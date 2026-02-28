#version 330 core
in vec3 vPos;
out vec4 FragColor;

void main() {
    // Normalize the position to get a direction vector
    vec3 dir = normalize(vPos);
    
    // Define the sky colors
    vec3 zenithColor = vec3(0.0, 0.4, 0.8);  // Deep Blue
    vec3 horizonColor = vec3(0.7, 0.85, 1.0); // Light Pale Blue
    
    // Calculate factor based on height (Y component)
    // We clamp it between 0.0 and 1.0
    float factor = clamp(dir.y, 0.0, 1.0);
    
    // Mix the colors based on the vertical factor
    vec3 finalColor = mix(horizonColor, zenithColor, factor);
    
    FragColor = vec4(finalColor, 1.0);
}