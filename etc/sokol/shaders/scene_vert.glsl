out vec4 position;
out vec4 light_position;
out vec3 normal;
out vec4 color;
out vec3 material;

void main() {
  vec4 pos4 = vec4(v_position, 1.0);
  gl_Position = u_mat_vp * i_mat_m * pos4;
  light_position = u_light_vp * i_mat_m * pos4;
  position = (i_mat_m * pos4);
  normal = (i_mat_m * vec4(v_normal, 0.0)).xyz;
  color = vec4(i_color, 0.0);
  material = i_material;
}
