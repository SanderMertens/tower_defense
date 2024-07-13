#define FLECS_COMPONENTS_GUI_IMPL

#include "flecs_components_gui.h"

ECS_CTOR(EcsText, ptr, {
    ptr->value = NULL;
})

ECS_DTOR(EcsText, ptr, {
    ecs_os_free(ptr->value);
    ptr->value = NULL;
})

ECS_COPY(EcsText, dst, src, {
    ecs_os_free(dst->value);
    dst->value = ecs_os_strdup(src->value);
})

ECS_MOVE(EcsText, dst, src, {
    ecs_os_free(dst->value);
    dst->value = src->value;
    src->value = NULL;
})

ECS_CTOR(EcsCanvas, ptr, {
    ecs_os_zeromem(ptr);
    ptr->ambient_light_ground_intensity = 1.0;
})

void FlecsComponentsGuiImport(
    ecs_world_t *world)
{
    ECS_MODULE(world, FlecsComponentsGui);
    ECS_IMPORT(world, FlecsComponentsGraphics);
    ECS_IMPORT(world, FlecsComponentsCglm);

    ecs_set_name_prefix(world, "Ecs");

    ECS_META_COMPONENT(world, EcsCanvas);
    ECS_META_COMPONENT(world, EcsText);
    ECS_META_COMPONENT(world, EcsFontSize);
    ECS_META_COMPONENT(world, EcsFontStyle);
    ECS_META_COMPONENT(world, EcsAlign);
    ECS_META_COMPONENT(world, EcsPadding);

    ecs_set_hooks(world, EcsCanvas, {
        .ctor = ecs_ctor(EcsCanvas),
    });

    ecs_set_hooks(world, EcsText, {
        .ctor = ecs_ctor(EcsText),
        .dtor = ecs_dtor(EcsText),
        .copy = ecs_copy(EcsText),
        .move = ecs_move(EcsText)
    });
}

