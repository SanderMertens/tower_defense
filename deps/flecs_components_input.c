#define FLECS_COMPONENTS_INPUT_IMPL

#include "flecs_components_input.h"
#include <string.h>

void FlecsComponentsInputImport(
    ecs_world_t *world)
{
    ECS_MODULE(world, FlecsComponentsInput);
    ECS_IMPORT(world, FlecsComponentsGraphics);

    ecs_set_name_prefix(world, "ecs");

    ECS_META_COMPONENT(world, ecs_key_state_t);
    ECS_META_COMPONENT(world, ecs_mouse_coord_t);
    ECS_META_COMPONENT(world, ecs_mouse_state_t);

    ecs_set_name_prefix(world, "Ecs");

    ECS_META_COMPONENT(world, EcsInput);
}

