#define FLECS_COMPONENTS_GUI_IMPL

#include "flecs_components_gui.h"

void FlecsComponentsGuiImport(
    ecs_world_t *world)
{
    ECS_MODULE(world, FlecsComponentsGui);
    ECS_IMPORT(world, FlecsComponentsGraphics);
    ECS_IMPORT(world, FlecsComponentsCglm);

    ecs_set_name_prefix(world, "Ecs");

    ECS_META_COMPONENT(world, EcsCanvas);
}

