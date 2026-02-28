import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import ctypes
from mesh_utils import ObjLoader, load_texture

# --- Configuration ---
RES_W, RES_H, FPS_LIMIT = 1280, 720, 165
FRAME_TIME = 1.0 / FPS_LIMIT

# Camera & Physics for a 1.2 Radius Planet
camera_pos = glm.vec3(0.0, 2.5, 0.0) # Spawn just above the "North Pole"
camera_velocity = glm.vec3(0.0)
camera_front = glm.vec3(0.0, -1.0, 0.0) # Look down at the planet initially
camera_up = glm.vec3(0.0, 0.0, 1.0)
first_mouse = True
last_x, last_y = RES_W / 2, RES_H / 2

PLANET_RADIUS = 1.2
PLAYER_HEIGHT = 0.3

def mouse_callback(window, xpos, ypos):
    global last_x, last_y, first_mouse, camera_front
    if first_mouse:
        last_x, last_y = xpos, ypos
        first_mouse = False

    sensitivity = 0.002
    dx, dy = xpos - last_x, ypos - last_y
    last_x, last_y = xpos, ypos

    # Dynamic basis for the sphere
    up = glm.normalize(camera_pos)
    right = glm.normalize(glm.cross(camera_front, up))

    # Rotation
    camera_front = glm.normalize(glm.rotate(camera_front, -dx * sensitivity, up))
    new_front = glm.normalize(glm.rotate(camera_front, -dy * sensitivity, right))
    
    # Pitch constraint relative to the local surface
    if abs(glm.dot(new_front, up)) < 0.98:
        camera_front = new_front

def process_input(window):
    global camera_pos, camera_velocity
    dt = FRAME_TIME
    
    # Tuned for small scale
    move_speed = 4.0
    gravity = 8.0
    damping = 10.0

    up = glm.normalize(camera_pos)
    right = glm.normalize(glm.cross(camera_front, up))
    forward = glm.normalize(glm.cross(up, right))

    # Gravity toward center
    camera_velocity -= up * gravity * dt

    # Movement
    accel = glm.vec3(0)
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS: accel += forward
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS: accel -= forward
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: accel -= right
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS: accel += right

    if glm.length(accel) > 0:
        camera_velocity += glm.normalize(accel) * move_speed * dt

    camera_pos += camera_velocity * dt
    camera_velocity -= camera_velocity * damping * dt

    # Surface collision
    dist = glm.length(camera_pos)
    target_dist = PLANET_RADIUS + PLAYER_HEIGHT
    if dist < target_dist:
        camera_pos = up * target_dist
        v_dot = glm.dot(camera_velocity, up)
        if v_dot < 0: camera_velocity -= v_dot * up

def main():
    if not glfw.init(): return
    window = glfw.create_window(RES_W, RES_H, "Mini Planet Explorer", None, None)
    if not window: return
    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_cursor_pos_callback(window, mouse_callback)

    obj_data, obj_v_count, _, _, tex_path = ObjLoader.load_model("planet.obj")
    tex_id = load_texture(tex_path) if tex_path else None

    # Shader with stronger local lighting
    v_src = """#version 330 core
    layout (location = 0) in vec3 pos;
    layout (location = 1) in vec3 norm;
    layout (location = 2) in vec3 col;
    layout (location = 3) in vec2 uv;
    uniform mat4 model, view, proj;
    out vec3 vNorm, vFragPos, vCol; out vec2 vUV;
    void main() {
        vFragPos = vec3(model * vec4(pos, 1.0));
        vNorm = mat3(transpose(inverse(model))) * norm;
        vCol = col; vUV = uv;
        gl_Position = proj * view * vec4(vFragPos, 1.0);
    }"""
    f_src = """#version 330 core
    in vec3 vNorm, vFragPos, vCol; in vec2 vUV;
    uniform sampler2D tex; uniform bool hasTex;
    out vec4 color;
    void main() {
        vec3 lightPos = vec3(5.0, 5.0, 5.0); // Closer light for a small planet
        vec3 n = normalize(vNorm);
        float diff = max(dot(n, normalize(lightPos - vFragPos)), 0.4); // Higher ambient floor
        vec3 base = hasTex ? texture(tex, vUV).rgb : vCol;
        color = vec4(base * diff, 1.0);
    }"""

    # Simple starfield skybox
    sky_v_src = """#version 330 core
    layout (location = 0) in vec3 pos;
    uniform mat4 view, proj;
    out vec3 vPos;
    void main() {
        vPos = pos;
        gl_Position = (proj * mat4(mat3(view)) * vec4(pos, 1.0)).xyww;
    }"""
    sky_f_src = """#version 330 core
    in vec3 vPos; out vec4 color;
    float hash(vec3 p) { return fract(sin(dot(p, vec3(12.9, 78.2, 45.1))) * 43758.5); }
    void main() {
        float star = pow(hash(floor(vPos * 50.0)), 400.0);
        color = vec4(vec3(star), 1.0) + vec4(0, 0, 0.05, 1);
    }"""

    p_shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(v_src, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(f_src, GL_FRAGMENT_SHADER))
    s_shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(sky_v_src, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(sky_f_src, GL_FRAGMENT_SHADER))

    # VAO setup
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, obj_data.nbytes, obj_data, GL_STATIC_DRAW)
    for i, size, offset in [(0,3,0), (1,3,12), (2,3,24), (3,2,36)]:
        glEnableVertexAttribArray(i); glVertexAttribPointer(i, size, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(offset))

    # Skybox VAO
    s_vao = glGenVertexArrays(1)
    glBindVertexArray(s_vao)
    svbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, svbo)
    sky_data = np.array([-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,-1,1,1,1,1,1,1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,1,1,1,1,1,1,-1,1,-1,-1,1,-1,1,-1,1,1,-1,1,1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,-1,-1,-1,1,1,-1,1], dtype=np.float32)
    glBufferData(GL_ARRAY_BUFFER, sky_data.nbytes, sky_data, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))

    glEnable(GL_DEPTH_TEST)

    while not glfw.window_should_close(window):
        process_input(window)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        up = glm.normalize(camera_pos)
        view = glm.lookAt(camera_pos, camera_pos + camera_front, up)
        # NEAR PLANE ADJUSTED TO 0.01 for small objects
        proj = glm.perspective(glm.radians(45.0), RES_W/RES_H, 0.01, 100.0)

        # Draw Planet
        glUseProgram(p_shader)
        glUniformMatrix4fv(glGetUniformLocation(p_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(p_shader, "proj"), 1, GL_FALSE, glm.value_ptr(proj))
        glUniformMatrix4fv(glGetUniformLocation(p_shader, "model"), 1, GL_FALSE, glm.value_ptr(glm.mat4(1.0)))
        glUniform1i(glGetUniformLocation(p_shader, "hasTex"), 1 if tex_id else 0)
        glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES, 0, obj_v_count)

        # Draw Skybox
        glDepthFunc(GL_LEQUAL)
        glUseProgram(s_shader)
        glUniformMatrix4fv(glGetUniformLocation(s_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(s_shader, "proj"), 1, GL_FALSE, glm.value_ptr(proj))
        glBindVertexArray(s_vao); glDrawArrays(GL_TRIANGLES, 0, 36)
        glDepthFunc(GL_LESS)

        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()

if __name__ == "__main__":
    main()