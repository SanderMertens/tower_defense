#include <tower_defense.h>
#include <vector>

using namespace flecs::components;
using Position = transform::Position3;
using Rotation = transform::Rotation3;
using Color = geometry::Color;
using Box = geometry::Box;

#define PI_2 ((float)(GLM_PI * 2))

// Game constants
static const float CameraAcceleration = 0.2;
static const float CameraDeceleration = 0.1;
static const float CameraMaxSpeed = 0.05;
static const float CameraDistance = 5.5;
static const float CameraHeight = 4.0;
static const float CameraLookatRadius = 2.0;

static const float EnemySize = 0.2;
static const float EnemySpeed = 1.5;
static const float EnemySpawnInterval = 1.0;

static const float TurretRotateSpeed = 2.0;

static const float TileSize = 1;
static const float TileHeight = 0.2;
static const float PathHeight = 0.1;
static const float TileSpacing = 0.0;
static const int TileCountX = 10;
static const int TileCountZ = 10;

// Direction vector
static const transform::Position2 dir[] = {
    {-1, 0},
    {0, -1},
    {1, 0},
    {0, 1},
};

// Grid helper class
template <typename T>
class grid {
public:
    grid(int width, int height)
        : m_width(width)
        , m_height(height) 
    { 
        // Create vector with X * Y elements
        for (int x = 0; x < width; x ++) {
            for (int y = 0; y < height; y ++) {
                m_values.push_back(T());
            }
        }
    }

    void set(int32_t x, int32_t y, T value) {
        m_values[y * m_width + x] = value;
    }

    const T operator()(int32_t x, int32_t y) {
        return m_values[y * m_width + x];
    }

private:
    int m_width;
    int m_height;
    std::vector<T> m_values;
};

// Global game data
struct Game {
    flecs::entity window;
    flecs::entity tile_prefab;
    flecs::entity path_prefab;
    flecs::entity enemy_prefab;
    flecs::entity turret_prefab;
    flecs::entity level;
};

// Level map and spawn point for enemies
struct Level {
    Level() {
        map = nullptr;
    }

    Level(grid<bool> *arg_map, transform::Position2 arg_spawn) {
        map = arg_map;
        spawn_point = arg_spawn;
    }

    grid<bool> *map;
    transform::Position2 spawn_point;
};

// Enemy tag
struct Enemy { };

// Movement direction for enemies
struct Direction {
    int value;
};

// Turret data
struct Turret { 
    flecs::entity head;
};

// Tracker for enemy target
struct Target {
    flecs::entity target;
};

// Query to find a target
struct TargetQuery {
    flecs::query<Position> query;
};

// Camera movement controller
struct CameraController {
    float r;
    float v;
};

// Calculate coordinate from position on the grid
float to_coord(float x) {
    return x * (TileSpacing + TileSize) - (TileSize / 2.0);
}

// Get cell on the grid from coordinate
float from_coord(float x) {
    return (x + (TileSize / 2.0)) / (TileSpacing + TileSize);
}

float to_x(float x) {
    return to_coord(x + 0.5) - to_coord((TileCountX / 2.0));
}

float to_z(float z) {
    return to_coord(z);
}

float from_x(float x) {
    return from_coord(x + to_coord((TileCountX / 2.0))) - 0.5;
}

float from_z(float z) {
    return from_coord(z);
}

float* to_vec3(Position& p) {
    return reinterpret_cast<float*>(&p);
}

// Init UI
void init_ui(flecs::world& ecs, flecs::entity game) {
    gui::Camera camera_data;
    camera_data.set_position(0, CameraHeight, 0);
    camera_data.set_lookat(0, 0, to_z(TileCountZ / 2));
    auto camera = ecs.entity("Camera")
        .set<gui::Camera>(camera_data)
        .set<CameraController>({-GLM_PI / 2, 0}); // Spherical camera coordinate

    gui::Window window_data;
    window_data.width = 1024;
    window_data.height = 800;
    auto window = ecs.entity().set<gui::Window>(window_data);

    gui::Canvas canvas_data;
    canvas_data.background_color = {0, 0, 0};
    canvas_data.camera = camera.id();
    window.set<gui::Canvas>(canvas_data);

    game.patch<Game>([window](Game& g) {
        g.window = window;
    });
}

// Init level
void init_level(flecs::world& ecs, flecs::entity game) {
    Game *g = game.get_mut<Game>();

    grid<bool> *path = new grid<bool>(TileCountX, TileCountZ);
    path->set(0, 1, true); path->set(1, 1, true); path->set(2, 1, true);
    path->set(3, 1, true); path->set(4, 1, true); path->set(5, 1, true);
    path->set(6, 1, true); path->set(7, 1, true); path->set(8, 1, true);
    path->set(8, 2, true); path->set(8, 3, true); path->set(7, 3, true);
    path->set(6, 3, true); path->set(5, 3, true); path->set(4, 3, true);
    path->set(3, 3, true); path->set(2, 3, true); path->set(1, 3, true);
    path->set(1, 4, true); path->set(1, 5, true); path->set(1, 6, true);
    path->set(1, 7, true); path->set(1, 8, true); path->set(2, 8, true);
    path->set(3, 8, true); path->set(4, 8, true); path->set(4, 7, true);
    path->set(4, 6, true); path->set(4, 5, true); path->set(5, 5, true);
    path->set(6, 5, true); path->set(7, 5, true); path->set(8, 5, true);
    path->set(8, 6, true); path->set(8, 7, true); path->set(7, 7, true);
    path->set(6, 7, true); path->set(6, 8, true); path->set(6, 9, true);
    path->set(7, 9, true); path->set(8, 9, true); path->set(9, 9, true);

    transform::Position2 spawn_point = {to_x(9), to_z(9)};
    
    g->level = ecs.entity()
        .set<Level>({path, spawn_point});

    for (int x = 0; x < TileCountX; x ++) {
        for (int z = 0; z < TileCountZ; z++) {
            float xc = to_x(x);
            float zc = to_z(z);

            auto t = ecs.entity();
            if (path[0](x, z)) {
                t.add_instanceof(g->path_prefab);
            } else {
                t.add_instanceof(g->tile_prefab);
            }

            t.set<Position>({xc, 0, zc});                
        }
    }

    ecs.entity()
        .add_instanceof(g->turret_prefab)
        .set<Position>({to_x(8), TileHeight / 2, to_z(4)}); 

    ecs.entity()
        .add_instanceof(g->turret_prefab)
        .set<Position>({to_x(7), TileHeight / 2, to_z(4)}); 

    ecs.entity()
        .add_instanceof(g->turret_prefab)
        .set<Position>({to_x(6), TileHeight / 2, to_z(4)}); 

    ecs.entity()
        .add_instanceof(g->turret_prefab)
        .set<Position>({to_x(3), TileHeight / 2, to_z(7)}); 

    ecs.entity()
        .add_instanceof(g->turret_prefab)
        .set<Position>({to_x(3), TileHeight / 2, to_z(6)});

    ecs.entity()
        .add_instanceof(g->turret_prefab)
        .set<Position>({to_x(3), TileHeight / 2, to_z(5)});        
}

// Init prefabs
void init_prefabs(flecs::world& ecs, flecs::entity game) {
    Game *g = game.get_mut<Game>();

    g->tile_prefab = ecs.prefab()
        .set<Color>({1.0, 1.0, 1.0})
        .set<Box>({TileSize, TileHeight, TileSize});

    g->path_prefab = ecs.prefab()
        .set<Color>({0.3, 0.3, 0.3})
        .set<Box>({TileSize + TileSpacing, PathHeight, TileSize + TileSpacing});

    g->enemy_prefab = ecs.prefab()
        .add<Enemy>()
        .set<Color>({1.0, 0.3, 0.3})
        .set<Box>({EnemySize, EnemySize, EnemySize});

    g->turret_prefab = ecs.prefab()
        .add<Turret>()
        .add<Target>()
        .set<TargetQuery>({ ecs.query<Position>("ANY:Enemy") })
        .add_owned<Turret>()
        .add_owned<Target>();

        // Feet
        ecs.prefab("TurretFeet")
            .add_childof(g->turret_prefab)
            .set<Color>({0.1, 0.1, 0.1})
            .set<Box>({0.3, 0.1, 0.3})
            .set<Position>({0, 0.05, 0});

        // Base
        ecs.prefab("TurretBase")
            .add_childof(g->turret_prefab)
            .set<Color>({0.5, 0.5, 0.5})
            .set<Box>({0.2, 0.3, 0.2})
            .set<Position>({0, 0.15, 0});

        // Head
        auto turret_head = ecs.prefab("TurretHead")
            .add_childof(g->turret_prefab)
            .set<Color>({0.35, 0.4, 0.3})
            .set<Box>({0.4, 0.2, 0.4})
            .set<Position>({0, 0.4, 0})
            .set<Rotation>({0, 0.0, 0});

            // Cannon
            ecs.prefab("TurretCannon")
                .add_childof(turret_head)
                .set<Color>({0.1, 0.1, 0.1})
                .set<Box>({0.4, 0.1, 0.1})
                .set<Position>({0.3, 0.0, 0});

    // When Turret is set, initialize it with the head child
    ecs.system<Turret>()
        .kind(flecs::OnSet)
        .each([](flecs::entity e, Turret& t) {
            t.head = e.lookup("TurretHead");
        });
}

// Move camera around with keyboard
void MoveCamera(flecs::iter& it) {
    auto input = it.column<const input::Input>(1);
    auto camera = it.column<gui::Camera>(2);
    auto ctrl = it.column<CameraController>(3);

    // Accelerate camera if keys are pressed
    if (input->keys[ECS_KEY_D].state) {
        ctrl->v -= CameraAcceleration * it.delta_time();
    }
    if (input->keys[ECS_KEY_A].state) {
        ctrl->v += CameraAcceleration * it.delta_time();
    }    

    // Decelerate camera each frame
    if (ctrl->v > 0) {
        ctrl->v -= CameraDeceleration * it.delta_time();
        if (ctrl->v < 0) {
            ctrl->v = 0;
        }
    }
    if (ctrl->v < 0) {
        ctrl->v += CameraDeceleration * it.delta_time();
        if (ctrl->v > 0) {
            ctrl->v = 0;
        }
    }

    // Make sure speed doesn't exceed limits
    ctrl->v = glm_clamp(ctrl->v, -CameraMaxSpeed, CameraMaxSpeed);

    // Update camera spherical coordinates
    ctrl->r += ctrl->v;

    camera->position[0] = cos(ctrl->r) * CameraDistance;
    camera->position[2] = sin(ctrl->r) * CameraDistance + to_z(TileCountZ / 2);

    camera->lookat[0] = cos(ctrl->r) * CameraLookatRadius;
    camera->lookat[2] = sin(ctrl->r) * CameraLookatRadius + to_z(TileCountZ / 2);
}

// Periodic system that spawns new enemies
void SpawnEnemy(flecs::iter& it) {
    auto ecs = it.world();
    auto g = it.column<const Game>(1);
    const Level* lvl = g->level.get<Level>();

    ecs.entity()
        .add_instanceof(g->enemy_prefab)
        .set<Direction>({0})
        .set<Position>({
            lvl->spawn_point.x, 0.3, lvl->spawn_point.y
        });
}

// Check if enemy needs to change direction
bool find_path(Position& p, Direction& d, const Level* lvl) {
    // Check if enemy is in center of tile
    float t_x = from_x(p.x);
    float t_y = from_z(p.z);
    int ti_x = (int)t_x;
    int ti_y = (int)t_y;
    float td_x = t_x - ti_x;
    float td_y = t_y - ti_y;

    // If enemy is in center of tile, decide where to go next
    if (td_x < 0.05 && td_y < 0.05) {
        grid<bool> *tiles = lvl->map;

        // Compute backwards direction so we won't try to go there
        int backwards = (d.value + 2) % 4;

        // Find a direction that the enemy can move to
        for (int i = 0; i < 3; i ++) {
            int n_x = ti_x + dir[d.value].x;
            int n_y = ti_y + dir[d.value].y;

            if (n_x >= 0 && n_x <= TileCountX) {
                if (n_y >= 0 && n_y <= TileCountZ) {
                    // Next tile is still on the grid, test if it's a path
                    if (tiles[0](n_x, n_y)) {
                        // Next tile is a path, so continue along current direction
                        return false;
                    }
                }
            }

            // Try next direction. Make sure not to move backwards
            do {
                d.value = (d.value + 1) % 4;
            } while (d.value == backwards);
        }

        // If enemy was not able to find a next direction, it reached the end
        return true;        
    }

    return false;
}

// Progress enemies along path
void MoveEnemy(flecs::iter& it, 
    flecs::column<Position> p,
    flecs::column<Direction> d) 
{
    auto ecs = it.world();
    auto g = it.column<const Game>(3);
    const Level* lvl = g->level.get<Level>();

    for (int i = 0; i < it.count(); i ++) {
        if (find_path(p[i], d[i], lvl)) {
            it.entity(i).destruct();
        } else {
            p[i].x += dir[d[i].value].x * EnemySpeed * it.delta_time();
            p[i].z += dir[d[i].value].y * EnemySpeed * it.delta_time();
        }
    }
}

// Find target for turret
void FindTarget(flecs::iter& it, 
    flecs::column<Target> target, 
    flecs::column<Position> p) 
{
    auto tq = it.column<const TargetQuery>(3);

    // For each turret, find closest enemy
    for (auto i : it) {
        // if (target[i].target) {
        //     // If turret already has a target, keep it until it becomes invalid
        //     continue;
        // }

        flecs::entity enemy;
        float distance = 0, min_distance = 0;
        auto pos = p[i];

        tq->query.each([&](
            flecs::entity e, Position& target_pos) 
        {
            distance = glm_vec3_distance(to_vec3(pos), to_vec3(target_pos));
            if (!min_distance || distance < min_distance) {
                min_distance = distance;
                enemy = e;
            }
        });

        target[i].target = enemy;
    }
}

float look_at(vec3 eye, vec3 dest) {
    vec3 diff;
    
    glm_vec3_sub(dest, eye, diff);
    float x = fabs(diff[0]), z = fabs(diff[2]);
    bool x_sign = diff[0] < 0, z_sign = diff[2] < 0;
    float r = atan(z / x);

    if (z_sign) {
        r += GLM_PI;
    }

    if (z_sign == x_sign) {
        r = -r + GLM_PI;
    }

    return r + GLM_PI;
}

float rotate_to(float cur, float target, float increment) {
    cur -= floor(cur / PI_2) * PI_2;
    target -= floor(target / PI_2) * PI_2;

    if (cur - target > GLM_PI) {
        cur -= PI_2;
    } else if (cur - target < -GLM_PI) {
        cur += PI_2;
    }
    
    if (cur > target) {
        cur -= increment;
        if (cur < target) {
            cur = target;
        }
    } else {
        cur += increment;
        if (cur > target) {
            cur = target;
        }
    }

    return cur;
}

// Aim turret at target
void AimTarget(flecs::iter& it, 
    flecs::column<Turret> turret, 
    flecs::column<Target> target, 
    flecs::column<Position> p) 
{
    for (auto i : it) {
        flecs::entity enemy = target[i].target;
        if (enemy) {
            Position target_p = enemy.get<Position>()[0];
            float angle = look_at(to_vec3(p[i]), to_vec3(target_p));
            Rotation *r = turret[i].head.get_mut<Rotation>();
            r->y = rotate_to(r->y, angle, TurretRotateSpeed * it.delta_time());
        }
    }
}

// Init systems
void init_systems(flecs::world& ecs) {
    ecs.system<>("MoveCamera", 
            "$:flecs.components.input.Input," 
            "Camera:flecs.components.gui.Camera,"
            "CameraController")
        .action(MoveCamera);

    ecs.system<>("SpawnEnemy", 
            "Game:Game")
        .interval(EnemySpawnInterval)
        .action(SpawnEnemy);

    ecs.system<Position, Direction>("MoveEnemy", 
            "Game:Game, ANY:Enemy")
        .action(MoveEnemy);

    ecs.system<Target, Position>("FindTarget",
            "SHARED:TargetQuery")
        .action(FindTarget);

    ecs.system<Turret, Target, Position>("AimTarget")
        .action(AimTarget);
}

int main(int argc, char *argv[]) {
    flecs::world ecs;

    ecs.enable_tracing(1);

    ecs.import<flecs::components::transform>();
    ecs.import<flecs::components::graphics>();
    ecs.import<flecs::components::geometry>();
    ecs.import<flecs::components::gui>();
    ecs.import<flecs::components::physics>();
    ecs.import<flecs::components::input>();
    ecs.import<flecs::systems::transform>();
    ecs.import<flecs::systems::sdl2>();
    ecs.import<flecs::systems::sokol>();

    auto game = ecs.entity("Game")
        .add<Game>();

    init_ui(ecs, game);

    init_prefabs(ecs, game);

    init_level(ecs, game);

    init_systems(ecs);

    while(ecs.progress()) { }
}
