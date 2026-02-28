#version 330 core
layout (location = 0) in vec3 pos;

out vec3 vPos;

uniform mat4 view;
uniform mat4 proj;

void main() {
    vPos = pos;
    // Remove translation from view matrix so skybox stays centered on camera
    mat4 viewNoTranslation = mat4(mat3(view));
    vec4 p = proj * viewNoTranslation * vec4(pos, 1.0);
    
    // Optimization: Force depth to 1.0 (the far plane)
    gl_Position = p.xyww;
}