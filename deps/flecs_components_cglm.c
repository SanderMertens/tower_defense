#include "flecs_components_cglm.h"

ECS_COMPONENT_DECLARE(vec3);
ECS_COMPONENT_DECLARE(vec4);

void FlecsComponentsCglmImport(
    ecs_world_t *world)
{
    ECS_MODULE(world, FlecsComponentsCglm);

    ecs_id(vec3) = ecs_array(world, {
        .entity = ecs_entity(world, {
            .name = "vec3",
            .symbol = "vec3",
            .id = ecs_id(vec3)
        }),
        .type = ecs_id(ecs_f32_t),
        .count = 3 
    });

    ecs_id(vec3) = ecs_array(world, {
        .entity = ecs_entity(world, {
            .name = "vec4",
            .symbol = "vec4",
            .id = ecs_id(vec4)
        }),
        .type = ecs_id(ecs_f32_t),
        .count = 4
    });
}

