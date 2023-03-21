
vec3 c = texture(hdr, uv).rgb;
vec3 b = texture(bloom, uv).rgb;
float b_clip = dot(vec3(0.333), b);
b = b + pow(b_clip, 3.0);
c = c + b;
c = pow(c, vec3(1.0 / gamma));
frag_color = vec4(c, 1.0);
