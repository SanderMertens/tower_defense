#define FLECS_COMPONENTS_GEOMETRY_IMPL
#include "flecs_components_geometry.h"

ECS_DECLARE(EcsGeometry);
ECS_COMPONENT_DECLARE(EcsStrokeColor);

ECS_COPY(EcsMesh, dst, src, {
    if (dst->vertices) {
        ecs_os_free(dst->vertices);
    }

    if (src->vertices) {
        size_t size = sizeof(vec3) * src->vertex_count;
        dst->vertices = ecs_os_malloc(size);
        dst->vertex_count = src->vertex_count;
        memcpy(dst->vertices, src->vertices, size);
    } else {
        dst->vertices = NULL;
        dst->vertex_count = 0;
    }
})

ECS_MOVE(EcsMesh, dst, src, {
    if (dst->vertices) {
        ecs_os_free(dst->vertices);
    }

    dst->vertices = src->vertices;
    dst->vertex_count = src->vertex_count;
    src->vertices = NULL;
    src->vertex_count = 0;
})

ECS_DTOR(EcsMesh, ptr, {
    ecs_os_free(ptr->vertices);
})

void FlecsComponentsGeometryImport(
    ecs_world_t *world)
{
    ECS_MODULE(world, FlecsComponentsGeometry);
    ECS_IMPORT(world, FlecsComponentsGraphics);

    ecs_set_name_prefix(world, "Ecs");

    ECS_META_COMPONENT(world, EcsDrawDistance);
    ECS_META_COMPONENT(world, EcsLine2);
    ECS_META_COMPONENT(world, EcsLine3);
    ECS_META_COMPONENT(world, EcsRectangle);
    ECS_META_COMPONENT(world, EcsStrokeWidth);
    ECS_COMPONENT_DEFINE(world, EcsStrokeColor);
    ECS_META_COMPONENT(world, EcsCornerRadius);
    ECS_META_COMPONENT(world, EcsBox);
    ECS_META_COMPONENT(world, EcsCircle);

    ECS_TAG_DEFINE(world, EcsGeometry);

    ecs_add_pair(world, ecs_id(EcsRectangle), EcsWith, EcsGeometry);
    ecs_add_pair(world, ecs_id(EcsBox), EcsWith, EcsGeometry);

    ecs_add_pair(world, ecs_id(EcsRectangle), EcsOnInstantiate, EcsInherit);
    ecs_add_pair(world, ecs_id(EcsBox), EcsOnInstantiate, EcsInherit);

    ecs_struct(world, {
        .entity = ecs_id(EcsStrokeColor),
        .members = {
            { "r", ecs_id(ecs_f32_t) },
            { "g", ecs_id(ecs_f32_t) },
            { "b", ecs_id(ecs_f32_t) }
        }
    });
}

