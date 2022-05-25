#version 450

#pragma shader_stage(vertex)

// Uniform block: PrivateUniforms
layout (std140, binding=0) uniform PrivateUniforms_vertex
{
    mat4 u_worldMatrix;
    mat4 u_viewProjectionMatrix;
    mat4 u_worldInverseTransposeMatrix;
};

// Inputs block: VertexInputs
layout (location = 0) in vec3 i_position;
layout (location = 1) in vec3 i_normal;
layout (location = 2) in vec2 i_texcoord_0;
layout (location = 3) in vec3 i_tangent;

layout (location = 0) out vec3 normalWorld;
layout (location = 1) out vec2 texcoord_0;
layout (location = 2) out vec3 tangentWorld;
layout (location = 3) out vec3 positionWorld;

void main()
{
    vec4 hPositionWorld = u_worldMatrix * vec4(i_position, 1.0);
    gl_Position = u_viewProjectionMatrix * hPositionWorld;
    normalWorld = normalize((u_worldInverseTransposeMatrix * vec4(i_normal, 0.0)).xyz);
    texcoord_0 = i_texcoord_0;
    tangentWorld = normalize((u_worldMatrix * vec4(i_tangent, 0.0)).xyz);
    // Omitted node 'N_mult_vector3'. Function already called in this scope.
    // Omitted node 'N_sub_vector3'. Function already called in this scope.
    // Omitted node 'N_divtilesize_vector3'. Function already called in this scope.
    // Omitted node 'N_multtilesize_vector3'. Function already called in this scope.
    // Omitted node 'N_img_vector3'. Function already called in this scope.
    // Omitted node 'N_mult_color3'. Function already called in this scope.
    // Omitted node 'N_sub_color3'. Function already called in this scope.
    // Omitted node 'N_divtilesize_color3'. Function already called in this scope.
    // Omitted node 'N_multtilesize_color3'. Function already called in this scope.
    // Omitted node 'N_img_color3'. Function already called in this scope.
    // Omitted node 'N_x_vector3'. Function already called in this scope.
    // Omitted node 'N_y_vector3'. Function already called in this scope.
    // Omitted node 'N_z_vector3'. Function already called in this scope.
    // Omitted node 'N_sw_vector3'. Function already called in this scope.
    // Omitted node 'N_x_vector3'. Function already called in this scope.
    // Omitted node 'N_y_vector3'. Function already called in this scope.
    // Omitted node 'N_z_vector3'. Function already called in this scope.
    // Omitted node 'N_sw_vector3'. Function already called in this scope.
    // Omitted node 'N_r_color3'. Function already called in this scope.
    // Omitted node 'N_g_color3'. Function already called in this scope.
    // Omitted node 'N_b_color3'. Function already called in this scope.
    // Omitted node 'N_sw_color3'. Function already called in this scope.
    // Omitted node 'N_r_color3'. Function already called in this scope.
    // Omitted node 'N_g_color3'. Function already called in this scope.
    // Omitted node 'N_b_color3'. Function already called in this scope.
    // Omitted node 'N_sw_color3'. Function already called in this scope.
    positionWorld = hPositionWorld.xyz;
}

