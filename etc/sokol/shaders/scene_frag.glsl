#include "etc/sokol/shaders/common.glsl"

uniform vec3 u_light_ambient;
uniform vec3 u_light_direction;
uniform vec3 u_light_color;
uniform vec3 u_eye_pos;
uniform float u_shadow_map_size;
uniform sampler2D shadow_map;
uniform float u_shadow_far;


in vec4 position;
in vec4 light_position;
in vec3 normal;
in vec4 color;
in vec3 material;
out vec4 frag_color;

const int pcf_count = 4;
const int pcf_samples = (2 * pcf_count + 1) * (2 * pcf_count + 1);
const float texel_c = 1.0;
const float soft_shadow_edge = 10.0;

float sampleShadow(sampler2D shadowMap, vec2 uv, float compare, float bias) {
  float depth = rgba_to_float(texture(shadowMap, vec2(uv.x, uv.y)));
  depth += bias;
  return step(compare, depth);
}

float sampleShadowPCF(sampler2D shadowMap, vec2 uv, float texel_size, float compare, float n_dot_l, float d) {
  float result = 0.0;
  float cos_theta = clamp(n_dot_l, 0.0, 1.0);
  float bias = 0.003 * tan(acos(cos_theta));
  
  float edge = clamp(d - (u_shadow_far - soft_shadow_edge), 0.0, soft_shadow_edge + 0.1);
  if (edge >= soft_shadow_edge) {
    return 1.0;
  }

  edge *= (1.0 / soft_shadow_edge);

  for (int x = -pcf_count; x <= pcf_count; x++) {
      for (int y = -pcf_count; y <= pcf_count; y++) {
          result += sampleShadow(shadowMap, uv + vec2(x, y) * texel_size * texel_c, compare, bias);
      }
  }

  result /= float(pcf_samples);
  result = mix(result, 1.0, edge);
  return result;
}

void main() {
  float specular_power = material.x;
  float shininess = max(material.y, 1.0);
  float emissive = material.z;
  vec3 l = normalize(u_light_direction);
  vec3 n = normalize(normal);
  float n_dot_l = dot(n, l);

  if (n_dot_l >= 0.0) {
    vec3 vd = u_eye_pos - position.xyz;
    vec3 v = normalize(vd);
    vec3 r = reflect(l, n);
    vec3 light_pos = light_position.xyz / light_position.w;
    vec2 sm_uv = (light_pos.xy + 1.0) * 0.5;
    float depth = light_position.z;
    float texel_size = 1.0 / u_shadow_map_size;
    float d = length(vd);
    float s = sampleShadowPCF(shadow_map, sm_uv, texel_size, depth, n_dot_l, d);

    s = max(s, emissive);
    float r_dot_v = max(dot(r, v), 0.0);
    float l_shiny = pow(r_dot_v * n_dot_l, shininess);
    vec3 l_specular = vec3(specular_power * l_shiny * u_light_color);
    vec3 l_diffuse = vec3(u_light_color) * n_dot_l;
    vec3 l_light = (u_light_ambient + s * l_diffuse);
    frag_color = vec4(max(vec3(emissive), l_light) * color.xyz + s * l_specular, 1.0);
  } else {
    vec3 light = emissive + clamp(1.0 - emissive, 0.0, 1.0) * (u_light_ambient);
    frag_color = vec4(light * color.xyz, 1.0);
  }
}
