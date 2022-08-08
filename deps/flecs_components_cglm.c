#include "flecs_components_cglm.h"

ECS_COMPONENT_DECLARE(vec3);

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
}

