#version 330 core

layout (location = 0) in vec2 InPos;
layout (location = 1) in vec2 InTexCoord;

// this name must match the name in the fragment shader
out vec2 TexCoord;

void main() {
    gl_Position = vec4(InPos.x, InPos.y, 0, 1.0);
    TexCoord = InTexCoord;
}