import numpy as np
import glm
import os
from PIL import Image
from OpenGL.GL import *
import OpenGL.GL.shaders

class ObjLoader:
    @staticmethod
    def load_material(file_path):
        materials = {}
        current_mat = None
        if not os.path.exists(file_path): return materials

        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.split()
                if not tokens: continue
                if tokens[0] == 'newmtl':
                    current_mat = tokens[1]
                    # Default to white so texture/color shows correctly
                    materials[current_mat] = {'kd': [1.0, 1.0, 1.0], 'texture': None}
                elif tokens[0] == 'Kd' and current_mat:
                    materials[current_mat]['kd'] = [float(x) for x in tokens[1:4]]
                elif tokens[0] == 'map_Kd' and current_mat:
                    materials[current_mat]['texture'] = os.path.join(os.path.dirname(file_path), tokens[1])
        return materials

    @staticmethod
    def load_model(file_path):
        cache_path = file_path + ".cache.npy"
        meta_path = file_path + ".meta.npy"
        base_path = os.path.dirname(file_path)

        # 1. Binary Loading (Fast Path)
        if os.path.exists(cache_path) and os.path.exists(meta_path):
            print(f"Loading cached binary: {cache_path}")
            final_data = np.load(cache_path)
            meta = np.load(meta_path)
            # Find the texture path again (it's not saved in binary)
            tex_path = None
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('mtllib '):
                        mats = ObjLoader.load_material(os.path.join(base_path, line.split()[1]))
                        tex_path = next((m['texture'] for m in mats.values() if m['texture']), None)
                        break
            return final_data, int(meta[0]), glm.vec3(*meta[1:4]), glm.vec3(*meta[4:7]), tex_path

        # 2. OBJ Parsing (Slow Path - First Time Only)
        print(f"Parsing OBJ: {file_path} (This will take a moment...)")
        temp_v, temp_vt, faces = [], [], []
        materials, current_mat = {}, None
        min_p, max_p = [float('inf')]*3, [float('-inf')]*3

        with open(file_path, 'r') as f:
            for line in f:
                tokens = line.split()
                if not tokens: continue
                if tokens[0] == 'v':
                    v = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
                    temp_v.append(v)
                    for i in range(3):
                        min_p[i] = min(min_p[i], v[i]); max_p[i] = max(max_p[i], v[i])
                elif tokens[0] == 'vt':
                    temp_vt.append([float(tokens[1]), float(tokens[2])])
                elif tokens[0] == 'mtllib':
                    materials = ObjLoader.load_material(os.path.join(base_path, tokens[1]))
                elif tokens[0] == 'usemtl':
                    current_mat = tokens[1]
                elif tokens[0] == 'f':
                    # Store (indices, current_material_name)
                    faces.append(([p.split('/') for p in tokens[1:]], current_mat))
                elif tokens[0] == 'Kd' and current_mat:
                    # Capture the RGB values (0.8 0.8 0.8 in your planet file)
                    materials[current_mat]['kd'] = [float(tokens[1]), float(tokens[2]), float(tokens[3])]

        # Progress bar setup
        total_faces = len(faces)
        total_tris = sum(len(f[0]) - 2 for f in faces)
        final_data = np.zeros(total_tris * 3 * 11, dtype=np.float32)
        
        vertices = np.array(temp_v, dtype=np.float32)
        uvs = np.array(temp_vt, dtype=np.float32) if temp_vt else None

        cursor = 0
        print("Building Mesh Data...")
        for idx, (face_tokens, m_name) in enumerate(faces):
            # Console Progress Bar
            while True:
                print(f"Progress: {int((idx/total_faces)*100)}%", end='\r')
                break

            mat = materials.get(m_name, {'kd': [1.0, 1.0, 1.0]})
            kd = mat['kd']
            
            # Triangulate
            for i in range(1, len(face_tokens) - 1):
                tri = [face_tokens[0], face_tokens[i], face_tokens[i+1]]
                # Normals
                v0, v1, v2 = vertices[int(tri[0][0])-1], vertices[int(tri[1][0])-1], vertices[int(tri[2][0])-1]
                norm = np.cross(v1 - v0, v2 - v0)
                n_len = np.linalg.norm(norm)
                if n_len > 0: norm /= n_len
                
                for p in tri:
                    v_idx = int(p[0]) - 1
                    t_idx = int(p[1]) - 1 if len(p) > 1 and p[1] else -1
                    # Fill pre-allocated array
                    final_data[cursor:cursor+3] = vertices[v_idx] # Pos
                    final_data[cursor+3:cursor+6] = norm          # Norm
                    final_data[cursor+6:cursor+9] = kd            # Color
                    if uvs is not None and t_idx != -1:
                        final_data[cursor+9:cursor+11] = uvs[t_idx] # UV
                    cursor += 11

        # Save Cache
        np.save(cache_path, final_data)
        np.save(meta_path, np.array([len(final_data)//11, *min_p, *max_p], dtype=np.float32))
        
        tex_path = next((m['texture'] for m in materials.values() if m['texture']), None)
        return final_data, len(final_data)//11, glm.vec3(*min_p), glm.vec3(*max_p), tex_path
        

def check_collision(camera_pos, mesh_min, mesh_max, radius=0.00001):
    if mesh_min is None or mesh_max is None: return False
    
    # AABB Collision logic
    closest_x = max(mesh_min.x, min(camera_pos.x, mesh_max.x))
    closest_y = max(mesh_min.y, min(camera_pos.y, mesh_max.y))
    closest_z = max(mesh_min.z, min(camera_pos.z, mesh_max.z))

    distance = glm.sqrt(
        (closest_x - camera_pos.x) ** 2 +
        (closest_y - camera_pos.y) ** 2 +
        (closest_z - camera_pos.z) ** 2
    )
    return distance < radius

def closest_point_on_triangle(p, a, b, c):
    """Finds the point on triangle abc closest to point p."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = glm.dot(ab, ap)
    d2 = glm.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0: return a
    
    bp = p - b
    d3 = glm.dot(ab, bp)
    d4 = glm.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3: return b
    
    cp = p - c
    d5 = glm.dot(ab, cp)
    d6 = glm.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6: return c
    
    # Check edge regions and interior
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab
        
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac
        
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)
        
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w

def check_exact_collision_fast(camera_pos, mesh_data, radius=0.2):
    # Reshape to (N, 3_vertices, 9_floats)
    tris = mesh_data.reshape(-1, 3, 11)
    A = tris[:, 0, :3]
    B = tris[:, 1, :3]
    C = tris[:, 2, :3]
    P = np.array([camera_pos.x, camera_pos.y, camera_pos.z], dtype=np.float32)

    # Broad phase: Quick sphere-to-triangle-bounding-box check
    t_min = np.minimum(np.minimum(A, B), C)
    t_max = np.maximum(np.maximum(A, B), C)
    
    # Check if point P is within distance 'radius' of each triangle's AABB
    in_range = np.all((P + radius >= t_min) & (P - radius <= t_max), axis=1)
    
    if not np.any(in_range):
        return False

    # Narrow phase: Only check triangles that passed the AABB test
    A, B, C = A[in_range], B[in_range], C[in_range]
    
    # Mathematical closest point check (Vectorized)
    # For performance, we simplify this to a distance check to the triangle center 
    # or vertices if the triangle count is still high.
    # Below is a simplified point-to-plane distance check for the subset
    v0, v1 = B - A, C - A
    normals = np.cross(v0, v1)
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    dist_to_plane = np.abs(np.sum((P - A) * normals, axis=1))
    
    # If the camera is very close to the plane of any triangle in range
    return np.any(dist_to_plane < radius)
def load_texture(path):
    try:
        img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
        img_data = img.convert("RGBA").tobytes()
        width, height = img.size

        tex_id = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0) # Ensure we are working on Unit 0
        glBindTexture(GL_TEXTURE_2D, tex_id)

        # Better wrapping for floor/wall textures
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        
        # Mipmapping prevents the "lines" and "shimmering" when far away
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D) # Generate mipmaps

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glGenerateMipmap(GL_TEXTURE_2D)
        
        print(f"Texture loaded successfully: {path} ({width}x{height})")
        return tex_id
    except Exception as e:
        print(f"Failed to load texture at {path}: {e}")
        return None