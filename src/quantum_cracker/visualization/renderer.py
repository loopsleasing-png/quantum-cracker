"""PyOpenGL 3D renderer for the Quantum Cracker simulation.

Renders the 78x78x78 voxel grid as a point cloud and 256 threads as
lines from center. Supports camera orbit/zoom, animation, and keyboard
controls.
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

try:
    import glfw
    from OpenGL.GL import (
        GL_ARRAY_BUFFER,
        GL_BLEND,
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_FALSE,
        GL_FLOAT,
        GL_FRAGMENT_SHADER,
        GL_LINES,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_POINTS,
        GL_PROGRAM_POINT_SIZE,
        GL_SRC_ALPHA,
        GL_STATIC_DRAW,
        GL_VERTEX_SHADER,
        glAttachShader,
        glBindBuffer,
        glBindVertexArray,
        glBlendFunc,
        glBufferData,
        glClear,
        glClearColor,
        glCompileShader,
        glCreateProgram,
        glCreateShader,
        glDeleteProgram,
        glDeleteShader,
        glDrawArrays,
        glEnable,
        glEnableVertexAttribArray,
        glGenBuffers,
        glGenVertexArrays,
        glGetProgramInfoLog,
        glGetShaderInfoLog,
        glGetUniformLocation,
        glLinkProgram,
        glShaderSource,
        glUniform1f,
        glUniformMatrix4fv,
        glUseProgram,
        glVertexAttribPointer,
    )

    HAS_GL = True
except ImportError:
    HAS_GL = False

import pyrr

if TYPE_CHECKING:
    from quantum_cracker.core.rip_engine import RipEngine
    from quantum_cracker.core.voxel_grid import SphericalVoxelGrid


def _compile_shader(source: str, shader_type: int) -> int:
    """Compile a GLSL shader from source."""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    info_log = glGetShaderInfoLog(shader)
    if info_log:
        raise RuntimeError(f"Shader compile error: {info_log.decode()}")
    return shader


def _create_program(vertex_src: str, fragment_src: str) -> int:
    """Create a shader program from vertex and fragment sources."""
    vs = _compile_shader(vertex_src, GL_VERTEX_SHADER)
    fs = _compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    info_log = glGetProgramInfoLog(program)
    if info_log:
        raise RuntimeError(f"Program link error: {info_log.decode()}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program


class QuantumRenderer:
    """PyOpenGL 3D renderer for the quantum cracker simulation.

    Keyboard controls:
        SPACE  -- pause/resume animation
        R      -- reset simulation
        +/-    -- adjust animation speed
        1      -- toggle voxel rendering
        2      -- toggle thread rendering
        ESC    -- close window
    """

    def __init__(
        self,
        grid: SphericalVoxelGrid,
        engine: RipEngine,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        if not HAS_GL:
            raise ImportError("PyOpenGL and glfw required for 3D rendering")

        self.grid = grid
        self.engine = engine
        self.width = width
        self.height = height

        # Camera state
        self.camera_distance: float = 3.0
        self.camera_azimuth: float = 45.0
        self.camera_elevation: float = 30.0

        # Mouse state
        self._mouse_pressed: bool = False
        self._last_mouse_x: float = 0.0
        self._last_mouse_y: float = 0.0

        # Animation state
        self.time: float = 0.0
        self.paused: bool = False
        self.animation_speed: float = 1.0

        # Render toggles
        self.show_voxels: bool = True
        self.show_threads: bool = True

        # GL handles
        self.window = None
        self.voxel_program: int = 0
        self.thread_program: int = 0
        self.voxel_vao: int = 0
        self.voxel_vbo: int = 0
        self.thread_vao: int = 0
        self.thread_vbo: int = 0
        self.voxel_count: int = 0
        self.thread_vertex_count: int = 0

    def initialize(self) -> None:
        """Set up GLFW window and OpenGL context."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        self.window = glfw.create_window(
            self.width, self.height, "Quantum Cracker", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_callback)

        glClearColor(0.05, 0.05, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_PROGRAM_POINT_SIZE)

        self._compile_shaders()
        self._setup_voxel_buffer()
        self._setup_thread_buffer()

    def _compile_shaders(self) -> None:
        """Compile all shader programs."""
        from quantum_cracker.visualization.shaders import (
            THREAD_FRAGMENT_SHADER,
            THREAD_VERTEX_SHADER,
            VOXEL_FRAGMENT_SHADER,
            VOXEL_VERTEX_SHADER,
        )

        self.voxel_program = _create_program(VOXEL_VERTEX_SHADER, VOXEL_FRAGMENT_SHADER)
        self.thread_program = _create_program(THREAD_VERTEX_SHADER, THREAD_FRAGMENT_SHADER)

    def _setup_voxel_buffer(self) -> None:
        """Create VBO for the voxel point cloud.

        Each vertex: [x, y, z, amplitude, energy, theta, phi] = 7 floats.
        Only include voxels above an energy threshold to avoid rendering
        empty space.
        """
        coords = self.grid.get_cartesian_coords()  # (N^3, 3)

        # Flatten state arrays
        amp_flat = self.grid.amplitude.ravel()
        energy_flat = self.grid.energy.ravel()

        # Build theta and phi for each voxel
        _, theta_grid, phi_grid = np.meshgrid(
            self.grid.r_coords, self.grid.theta_coords, self.grid.phi_coords,
            indexing="ij",
        )
        theta_flat = theta_grid.ravel()
        phi_flat = phi_grid.ravel()

        # Filter: only voxels with energy above 10th percentile
        threshold = np.percentile(energy_flat[energy_flat > 0], 10) if np.any(energy_flat > 0) else 0.0
        mask = energy_flat > threshold

        # Build vertex data: [x, y, z, amplitude, energy, theta, phi]
        vertices = np.column_stack([
            coords[mask],
            amp_flat[mask],
            energy_flat[mask],
            theta_flat[mask],
            phi_flat[mask],
        ]).astype(np.float32)

        self.voxel_count = len(vertices)

        self.voxel_vao = glGenVertexArrays(1)
        self.voxel_vbo = glGenBuffers(1)

        glBindVertexArray(self.voxel_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.voxel_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        stride = 7 * 4  # 7 floats * 4 bytes
        # position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # amplitude
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        # energy
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(16))
        glEnableVertexAttribArray(2)
        # theta
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(20))
        glEnableVertexAttribArray(3)
        # phi
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(4)

        glBindVertexArray(0)

    def _setup_thread_buffer(self) -> None:
        """Create VBO for thread lines.

        Each thread is 2 vertices (origin -> direction * scale).
        Each vertex: [x, y, z, visible] = 4 floats.
        """
        scale = 1.2  # Visual length of threads
        origin = np.zeros(3, dtype=np.float32)

        vertices = []
        for i in range(self.engine.num_threads):
            d = self.engine.directions[i].astype(np.float32) * scale
            vis = 1.0 if self.engine.visible[i] else 0.0
            vertices.append(np.concatenate([origin, [vis]]))
            vertices.append(np.concatenate([d, [vis]]))

        vertex_data = np.array(vertices, dtype=np.float32)
        self.thread_vertex_count = len(vertex_data)

        self.thread_vao = glGenVertexArrays(1)
        self.thread_vbo = glGenBuffers(1)

        glBindVertexArray(self.thread_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.thread_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        stride = 4 * 4  # 4 floats * 4 bytes
        # position
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # visible
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        glBindVertexArray(0)

    def _update_thread_buffer(self) -> None:
        """Update thread visibility data in the VBO."""
        scale = 1.2
        origin = np.zeros(3, dtype=np.float32)

        vertices = []
        for i in range(self.engine.num_threads):
            d = self.engine.directions[i].astype(np.float32) * scale
            vis = 1.0 if self.engine.visible[i] else 0.0
            vertices.append(np.concatenate([origin, [vis]]))
            vertices.append(np.concatenate([d, [vis]]))

        vertex_data = np.array(vertices, dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.thread_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

    def _build_view_matrix(self) -> NDArray[np.float32]:
        """Build camera view matrix from azimuth/elevation/distance."""
        az = np.radians(self.camera_azimuth)
        el = np.radians(self.camera_elevation)
        eye = np.array([
            self.camera_distance * np.cos(el) * np.sin(az),
            self.camera_distance * np.sin(el),
            self.camera_distance * np.cos(el) * np.cos(az),
        ], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return pyrr.matrix44.create_look_at(eye, target, up)

    def _build_projection_matrix(self) -> NDArray[np.float32]:
        """Build perspective projection matrix."""
        return pyrr.matrix44.create_perspective_projection_matrix(
            45.0, self.width / self.height, 0.1, 100.0
        )

    def render_frame(self) -> None:
        """Render one frame."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        view = self._build_view_matrix()
        projection = self._build_projection_matrix()

        if self.show_voxels and self.voxel_count > 0:
            glUseProgram(self.voxel_program)
            glUniformMatrix4fv(
                glGetUniformLocation(self.voxel_program, "view"),
                1, GL_FALSE, view,
            )
            glUniformMatrix4fv(
                glGetUniformLocation(self.voxel_program, "projection"),
                1, GL_FALSE, projection,
            )
            glUniform1f(
                glGetUniformLocation(self.voxel_program, "time"),
                self.time,
            )
            glBindVertexArray(self.voxel_vao)
            glDrawArrays(GL_POINTS, 0, self.voxel_count)

        if self.show_threads and self.thread_vertex_count > 0:
            glUseProgram(self.thread_program)
            glUniformMatrix4fv(
                glGetUniformLocation(self.thread_program, "view"),
                1, GL_FALSE, view,
            )
            glUniformMatrix4fv(
                glGetUniformLocation(self.thread_program, "projection"),
                1, GL_FALSE, projection,
            )
            glBindVertexArray(self.thread_vao)
            glDrawArrays(GL_LINES, 0, self.thread_vertex_count)

        glBindVertexArray(0)
        glfw.swap_buffers(self.window)

    def run(self) -> None:
        """Main render loop."""
        self.initialize()

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            if not self.paused:
                dt = 0.016 * self.animation_speed  # ~60fps
                self.time += dt
                self.engine.step(dt=dt)
                self._update_thread_buffer()

            self.render_frame()

        self._cleanup()

    def _key_callback(self, window, key, scancode, action, mods) -> None:  # type: ignore[no-untyped-def]
        """Handle keyboard input."""
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            self.paused = not self.paused
        elif key == glfw.KEY_R:
            self.time = 0.0
            self.engine.initialize_random()
        elif key == glfw.KEY_EQUAL:  # +
            self.animation_speed = min(self.animation_speed * 1.5, 10.0)
        elif key == glfw.KEY_MINUS:
            self.animation_speed = max(self.animation_speed / 1.5, 0.1)
        elif key == glfw.KEY_1:
            self.show_voxels = not self.show_voxels
        elif key == glfw.KEY_2:
            self.show_threads = not self.show_threads

    def _mouse_button_callback(self, window, button, action, mods) -> None:  # type: ignore[no-untyped-def]
        """Handle mouse button for orbit."""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._mouse_pressed = action == glfw.PRESS
            if self._mouse_pressed:
                self._last_mouse_x, self._last_mouse_y = glfw.get_cursor_pos(window)

    def _scroll_callback(self, window, xoff, yoff) -> None:  # type: ignore[no-untyped-def]
        """Zoom in/out."""
        self.camera_distance = max(0.5, self.camera_distance - yoff * 0.3)

    def _cursor_callback(self, window, xpos, ypos) -> None:  # type: ignore[no-untyped-def]
        """Orbit camera on drag."""
        if not self._mouse_pressed:
            return
        dx = xpos - self._last_mouse_x
        dy = ypos - self._last_mouse_y
        self.camera_azimuth += dx * 0.3
        self.camera_elevation = max(-89, min(89, self.camera_elevation + dy * 0.3))
        self._last_mouse_x = xpos
        self._last_mouse_y = ypos

    def _cleanup(self) -> None:
        """Destroy window and terminate GLFW."""
        if self.voxel_program:
            glDeleteProgram(self.voxel_program)
        if self.thread_program:
            glDeleteProgram(self.thread_program)
        glfw.terminate()
