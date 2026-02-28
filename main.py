import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math
import ctypes
import sys
from mesh_utils import ObjLoader, check_collision, check_exact_collision_fast, load_texture
import os
import json
from pyrr import matrix44


def load_scene(path):
    with open(path, 'r') as f:
        return json.load(f)

scene = load_scene('scene.json')
# --- Configuration Inputs (Run before Window creation) ---
print("--- Renderer Configuration ---")
try:
    RES_W = int(input("Enter Width (e.g. 1280): ") or 1280)
    RES_H = int(input("Enter Height (e.g. 720): ") or 720)
    FPS_LIMIT = int(input("Enter FPS Limit (e.g. 60): ") or 165)
except ValueError:
    RES_W, RES_H, FPS_LIMIT = 1280, 720, 60

FRAME_TIME = 1.0 / FPS_LIMIT

# Camera State
camera_pos = glm.vec3(0.0, 10, 0.0) 
#camera_front = glm.vec3(0.0, 0.0, 1.0)
camera_up = glm.vec3(0.0, 1.0, 0.0) 
first_mouse = True
last_x, last_y = RES_W / 2, RES_H / 2
camera_velocity = glm.vec3(0)
camera_forward = glm.vec3(0, 0, 1)  # tangent-space forward

yaw = 0.0
pitch = 0.0
loaded_meshes = {} 
scene_data = []


# Globals for Collision
obj_min, obj_max = None, None
planet_radius = 0.0
PLAYER_HEIGHT = 0.15
SKY_RADIUS = 40.0
# Set to True to force a fullscreen cyan debug sky (helps diagnose draw/state issues)
DEBUG_FORCE_CYAN = False
star_debug_printed = False

def mouse_callback(window, xpos, ypos):
    global last_x, last_y, first_mouse, yaw, pitch
    global camera_pos

    if first_mouse:
        last_x, last_y = xpos, ypos
        first_mouse = False

    # Reduce yaw sensitivity near the planet poles to avoid large jumps
    sensitivity = 0.002
    try:
        up = glm.normalize(camera_pos)
        pole_factor = abs(glm.dot(up, glm.vec3(0,1,0)))
        # scale sensitivity down as we approach the pole (clamp min)
        sensitivity *= max(0.05, 1.0 - (pole_factor - 0.9) * 5.0) if pole_factor > 0.9 else 1.0
    except Exception:
        pass
    dx = xpos - last_x
    dy = ypos - last_y
    last_x, last_y = xpos, ypos

    # Update global angles
    yaw -= dx * sensitivity
    pitch -= dy * sensitivity

    # Clamp pitch to prevent flipping (approx 89 degrees)
    pitch = max(-1.5, min(1.5, pitch))

if obj_min is not None and obj_max is not None:
    # Better approximate planet radius from axis-aligned bounds by taking
    # the maximum absolute component. This avoids overestimating radius
    # when min/max are corners (which have length ~= 1.73*r for a sphere).
    planet_radius = max(
        abs(obj_min.x), abs(obj_min.y), abs(obj_min.z),
        abs(obj_max.x), abs(obj_max.y), abs(obj_max.z)
    )
    # Ensure the camera starts at or above the surface by default
    if glm.length(camera_pos) < planet_radius + PLAYER_HEIGHT:
        camera_pos = glm.normalize(camera_pos) * (planet_radius + PLAYER_HEIGHT)

def process_input(window):
    global loaded_meshes
    global camera_pos, camera_velocity
    global camera_forward, obj_min, obj_max, obj_data, yaw, pitch, planet_radius
    global scene_data
    # Free-fly camera controls (no gravity)
    dt = FRAME_TIME
    fly_speed = 8.0
    collision_radius = 0.2

    # Use instantaneous movement (no inertia)
    move_dir = glm.vec3(0, 0, 0)

    # Movement basis from camera_forward and world up
    world_up = glm.vec3(0, 1, 0)
    move_forward = camera_forward
    if glm.length(move_forward) < 1e-6:
        move_forward = glm.vec3(0, 0, 1)
    move_right = glm.cross(move_forward, world_up)
    if glm.length(move_right) < 1e-6:
        move_right = glm.vec3(1, 0, 0)
    else:
        move_right = glm.normalize(move_right)

    move_up = world_up

    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS: move_dir += move_forward
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS: move_dir -= move_forward
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: move_dir -= move_right
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS: move_dir += move_right
    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS: move_dir += move_up
    if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS: move_dir -= move_up
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS: glfw.terminate(); sys.exit(0)

    if glm.length(move_dir) > 0:
        move_dir = glm.normalize(move_dir)
        disp = move_dir * fly_speed * dt
        prev_pos = glm.vec3(camera_pos)
        camera_pos += disp

        with open('scene.json', 'r') as f:
            scene_data = json.load(f)

        # Collision: simple check with mesh; if colliding, try sliding or revert
        for obj in scene_data:
            mesh = loaded_meshes[obj['mesh']]
            pos = glm.vec3(*obj['position'])
            scale = glm.vec3(*obj['scale'])

            # Transform camera into the object's local space
            # LocalPos = (WorldPos - ObjectPos) / Scale
            local_cam_pos = (camera_pos - pos)
            local_cam_pos.x /= scale.x
            local_cam_pos.y /= scale.y
            local_cam_pos.z /= scale.z

            # Use the local position against the original mesh data
            # Scale the radius down to match the local space
            local_radius = 0.2 / max(scale.x, scale.y, scale.z)
            
            if check_exact_collision_fast(local_cam_pos, mesh['data'], radius=local_radius):
                # Collision detected! Revert movement or slide
                camera_pos = prev_pos 
                break

    # No damping for free-fly camera; keep position debug print



def main():
    global obj_min, obj_max, camera_forward
    global star_debug_printed
    global loaded_meshes, scene_data
    
    if not glfw.init(): return
    
    # Critical: Set these hints before creating the window to fix NullFunctionError
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

    window = glfw.create_window(RES_W, RES_H, "3D Renderer - Debug Mode", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_cursor_pos_callback(window, mouse_callback)

    # 1. Load data inside main after context is created
    

    # Shader Setup
# Updated Vertex Shader
    v_src = """#version 330 core
        layout (location = 0) in vec3 pos;
        layout (location = 1) in vec3 normal;
        layout (location = 2) in vec3 col;
        layout (location = 3) in vec2 uv; // Added UV input

        uniform mat4 model; uniform mat4 view; uniform mat4 proj;

        out vec3 FragPos;
        out vec3 Normal;
        out vec3 vCol;
        out vec2 vUV; // Pass to fragment shader

        void main() {
            FragPos = vec3(model * vec4(pos, 1.0));
            Normal = mat3(transpose(inverse(model))) * normal;
            vCol = col;
            vUV = uv; 
            gl_Position = proj * view * vec4(FragPos, 1.0);
        }"""

    # Updated Fragment Shader (Simple Diffuse Lighting)
    f_src = """#version 330 core
        in vec3 FragPos, Normal, vCol;
        in vec2 vUV; // Received from vertex shader
        out vec4 color;
        
        uniform sampler2D tex;
        uniform bool hasTexture;
        
        void main() {
            vec3 lightDir = normalize(vec3(5, 10, 5) - FragPos);
            float diff = max(dot(normalize(Normal), lightDir), 0.3); // Ambient 0.3
            
            vec4 texSample = hasTexture ? texture(tex, vUV) : vec4(1.0);
            
            // Multiply light * vertex color * texture
            color = vec4(diff * vCol * texSample.rgb, 1.0);
        }"""
    sky_v_src = """#version 330 core
        layout (location = 0) in vec3 pos;
        out vec3 TexCoords;
        uniform mat4 view;
        uniform mat4 proj;
        void main() {
            TexCoords = pos;
            vec4 p = proj * mat4(mat3(view)) * vec4(pos, 1.0);
            gl_Position = p.xyww; // Force skybox to background
    }"""

    sky_f_src = """#version 330 core
        out vec4 FragColor;
        in vec3 TexCoords;
        uniform samplerCube skybox;
        void main() {
            FragColor = texture(skybox, TexCoords);
    }"""

    try:
        shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(v_src, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(f_src, GL_FRAGMENT_SHADER))
    except Exception as e:
        print(f"SHADER ERROR: {e}")
        return
    # Skybox shader (procedural starfield)
    # Try loading from external shader files `sky.vert` / `sky.frag` next to this script.
    sky_vert_path = os.path.join(os.path.dirname(__file__), "sky.vert")
    sky_frag_path = os.path.join(os.path.dirname(__file__), "sky.frag")
    if os.path.exists(sky_vert_path) and os.path.exists(sky_frag_path):
        with open(sky_vert_path, 'r', encoding='utf-8') as f:
            sky_v_src = f.read()
        with open(sky_frag_path, 'r', encoding='utf-8') as f:
            sky_f_src = f.read()
    else:
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
    try:
        sky_shader = OpenGL.GL.shaders.compileProgram(
            OpenGL.GL.shaders.compileShader(sky_v_src, GL_VERTEX_SHADER),
            OpenGL.GL.shaders.compileShader(sky_f_src, GL_FRAGMENT_SHADER))
    except Exception as e:
        print(f"SKY SHADER ERROR: {e}")
        print("--- sky.vert (first 200 chars):\n", sky_v_src[:200])
        print("--- sky.frag (first 200 chars):\n", sky_f_src[:200])
        sky_shader = None
    else:
        print(f"SKY SHADER compiled: {sky_shader is not None}")
    # Star VBO layer removed; rely on procedural `sky.frag` for stars.
    star_shader = None
    # Fullscreen cyan debug shader removed (debugging done).
    # Manual star shader removed; using only the VBO-based star renderer.
    # NDC overlay removed (debug overlay not required)
    stride = 44
    # 2. Setup GPU Buffers only after context is current
    # Load the scene configuration
    with open('scene.json', 'r') as f:
        scene_data = json.load(f)

    loaded_meshes = {}

    # Dictionary to store unique mesh data { "filename.obj": (vao, vertex_count) }
    for entry in scene_data:
        m_path = entry['mesh']
        if m_path not in loaded_meshes:
            print(f"Initializing mesh: {m_path}")
            data, v_count, m_min, m_max, tex_path = ObjLoader.load_model(m_path)
            
            # GPU Buffer Setup
            vao = glGenVertexArrays(1)
            vbo = glGenBuffers(1)
            glBindVertexArray(vao)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
            
            # Attribute Pointers (Stride 44)
            for i, offset in enumerate([0, 12, 24, 36]):
                glEnableVertexAttribArray(i)
                glVertexAttribPointer(i, 3 if i<3 else 2, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(offset))
                
            # Load texture if it exists
            t_id = load_texture(tex_path) if tex_path else None
            
            loaded_meshes[m_path] = {
                "data": data, 
                "vao": vao, 
                "v_count": v_count, 
                "tex": t_id,
                "min": m_min,
                "max": m_max
            }

    # Skybox cube (36 verts) - scaled to sit near the far plane
    sky_vertices = np.array([
        -1,-1,-1,  1,-1,-1,  1, 1,-1,  1, 1,-1, -1, 1,-1, -1,-1,-1,
        -1,-1, 1,  1,-1, 1,  1, 1, 1,  1, 1, 1, -1, 1, 1, -1,-1, 1,
        -1, 1, 1, -1, 1,-1, -1,-1,-1, -1,-1,-1, -1,-1, 1, -1, 1, 1,
         1, 1, 1,  1, 1,-1,  1,-1,-1,  1,-1,-1,  1,-1, 1,  1, 1, 1,
        -1,-1,-1,  1,-1,-1,  1,-1, 1,  1,-1, 1, -1,-1, 1, -1,-1,-1,
        -1, 1,-1,  1, 1,-1,  1, 1, 1,  1, 1, 1, -1, 1, 1, -1, 1,-1
    ], dtype=np.float32) * SKY_RADIUS
    sky_vao = glGenVertexArrays(1)
    sky_vbo = glGenBuffers(1)
    glBindVertexArray(sky_vao)
    glBindBuffer(GL_ARRAY_BUFFER, sky_vbo)
    glBufferData(GL_ARRAY_BUFFER, sky_vertices.nbytes, sky_vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # VBO star layer removed; procedural sky will handle background stars.

    # Setup Grid
    grid_data = []
    for i in range(-10, 11):
        grid_data.extend([i, 0, -10, 0.5, 0.5, 0.5, i, 0, 10, 0.5, 0.5, 0.5])
        grid_data.extend([-10, 0, i, 0.5, 0.5, 0.5, 10, 0, i, 0.5, 0.5, 0.5])
    grid_data = np.array(grid_data, dtype=np.float32)

    grid_vao = glGenVertexArrays(1)
    grid_vbo = glGenBuffers(1)
    glBindVertexArray(grid_vao)
    glBindBuffer(GL_ARRAY_BUFFER, grid_vbo)
    glBufferData(GL_ARRAY_BUFFER, grid_data.nbytes, grid_data, GL_STATIC_DRAW)

    # Position (Location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # Normals (Location 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)

    # Colors (Location 2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
    glEnableVertexAttribArray(2)

    # UVs (Location 3)
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(36))
    glEnableVertexAttribArray(3)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    # (removed fullscreen cyan VAO - debug background disabled)
    
    last_time = glfw.get_time()
    frame_count = 0
    prev_frame_time = glfw.get_time()

    glUseProgram(shader)
    # This tells the shader that 'tex' refers to GL_TEXTURE0

    tex_location = glGetUniformLocation(shader, "tex")
    glUniform1i(tex_location, 0) 

    has_tex_location = glGetUniformLocation(shader, "hasTexture")

    while not glfw.window_should_close(window):
        curr_time = glfw.get_time()
        
        if curr_time - prev_frame_time < FRAME_TIME:
            continue
        prev_frame_time = curr_time

        frame_count += 1
        if curr_time - last_time >= 1.0:
            glfw.set_window_title(window, f"3D Renderer | FPS: {frame_count}")
            frame_count = 0
            last_time = curr_time

        # Compute camera forward from yaw/pitch (free camera)
        fx = math.sin(yaw) * math.cos(pitch)
        fy = math.sin(pitch)
        fz = math.cos(yaw) * math.cos(pitch)
        camera_forward = glm.normalize(glm.vec3(fx, fy, fz))

        process_input(window)

        glClearColor(0.0, 0.0, 0.001, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # View Matrix (free camera uses world up)
        view = glm.lookAt(camera_pos, camera_pos + camera_forward, glm.vec3(0,1,0))
        proj = glm.perspective(glm.radians(45.0), RES_W/RES_H, 0.1, 100.0)

        # Draw skybox first (if available) -- allow forcing a fullscreen debug sky
        if DEBUG_FORCE_CYAN or (sky_shader is not None):
            glDepthFunc(GL_LEQUAL)
            glDepthMask(GL_FALSE)
            # Ensure skybox faces aren't culled
            prev_cull = glIsEnabled(GL_CULL_FACE)
            if prev_cull:
                glDisable(GL_CULL_FACE)
            # Draw fullscreen debug pass only when explicitly enabled
            # fullscreen cyan debug draw disabled

            # star draw remains handled after this sky block

            # Then draw procedural box over it (if compiled)
            if sky_shader is not None:
                glUseProgram(sky_shader)
                glUniformMatrix4fv(glGetUniformLocation(sky_shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
                glUniformMatrix4fv(glGetUniformLocation(sky_shader, "proj"), 1, GL_FALSE, glm.value_ptr(proj))
                glBindVertexArray(sky_vao)
                glDrawArrays(GL_TRIANGLES, 0, 36)
            # restore culling
            if prev_cull:
                glEnable(GL_CULL_FACE)
            glDepthMask(GL_TRUE)
            glDepthFunc(GL_LESS)
        # VBO star layer removed — one-time debug print instead
        if not star_debug_printed:
            print("Star VBO layer removed; procedural sky active.")
            star_debug_printed = True
        # (removed manual/NDC overlay debug draws)
        glUseProgram(shader)
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(shader, "proj"), 1, GL_FALSE, glm.value_ptr(proj))

        # Toggle face culling when camera is extremely close to surface
        # This prevents the planet from disappearing when the camera is inside
        # or extremely near the geometry (camera interior leads to backfaces).
        cam_dist = glm.length(camera_pos)
        if planet_radius > 0.0 and cam_dist < planet_radius + 0.5:
            glDisable(GL_CULL_FACE)
        else:
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)

        # Draw Model (Chair)
        # Draw all objects defined in scene.json
        model_loc = glGetUniformLocation(shader, "model")
        mesh_assets = {}
        for obj in scene_data:
            mesh = loaded_meshes[obj['mesh']]
            
            # 1. Setup Matrices
            model = glm.mat4(1.0)
            model = glm.translate(model, glm.vec3(*obj['position']))
            model = glm.scale(model, glm.vec3(*obj['scale']))
            glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
            
            # 2. Bind Texture
            if mesh['tex']:
                glBindTexture(GL_TEXTURE_2D, mesh['tex'])
                glUniform1i(glGetUniformLocation(shader, "hasTexture"), 1)
            else:
                glUniform1i(glGetUniformLocation(shader, "hasTexture"), 0)
                
            # 3. Draw
            glBindVertexArray(mesh['vao'])
            glDrawArrays(GL_TRIANGLES, 0, mesh['v_count'])

        # Draw Grid
        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(glm.mat4(1.0)))
        glBindVertexArray(grid_vao)
        glDrawArrays(GL_LINES, 0, len(grid_data) // 6)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()