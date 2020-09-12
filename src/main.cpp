#include <tower_defense.h>
#include <vector>

using namespace flecs::components;

// Game constants
static const float EnemySize = 0.5;
static const float TileSize = 1;
static const float TileHeight = 0.4;
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

// Game components
struct Game {
    flecs::entity window;
    flecs::entity tile_prefab;
    flecs::entity path_prefab;
    flecs::entity enemy_prefab;
    flecs::entity level;
};

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

struct Enemy { };

struct Direction {
    int value;
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

// Init UI
void init_ui(flecs::world& ecs, flecs::entity game) {
    gui::Camera camera_data;
    camera_data.set_position(0, 5, 0);
    camera_data.set_lookat(0, 0, 2.5);
    auto camera = ecs.entity().set<gui::Camera>(camera_data);

    gui::Window window_data;
    window_data.width = 800;
    window_data.height = 600;
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

            t.set<transform::Position3>({xc, 0, zc});                
        }
    }
}

// Init prefabs
void init_prefabs(flecs::world& ecs, flecs::entity game) {
    Game *g = game.get_mut<Game>();

    g->tile_prefab = ecs.prefab()
        .set<geometry::Color>({1.0, 1.0, 1.0})
        .set<geometry::Box>({TileSize, TileHeight, TileSize});

    g->path_prefab = ecs.prefab()
        .set<geometry::Color>({0.3, 0.3, 0.3})
        .set<geometry::Box>({TileSize, PathHeight, TileSize});

    g->enemy_prefab = ecs.prefab()
        .add<Enemy>()
        .set<geometry::Color>({1.0, 0.3, 0.3})
        .set<geometry::Box>({EnemySize, EnemySize, EnemySize});        
}

// Periodic system that spawns new enemies
void SpawnEnemy(flecs::iter& it) {
    auto ecs = it.world();
    auto g = it.column<const Game>(1);
    const Level* lvl = g->level.get<Level>();

    ecs.entity()
        .add_instanceof(g->enemy_prefab)
        .set<Direction>({0})
        .set<transform::Position3>({
            lvl->spawn_point.x, 0.3, lvl->spawn_point.y
        });
}

// Check if enemy needs to change direction
void find_path(transform::Position3& p, Direction& d, const Level* lvl) {
    // Check if enemy is in center of tile
    float t_x = from_x(p.x);
    float t_y = from_z(p.z);
    int ti_x = (int)t_x;
    int ti_y = (int)t_y;
    float td_x = t_x - ti_x;
    float td_y = t_y - ti_y;

    // If enemy is in center of tile, decide where to go next
    if (td_x < 0.1 && td_y < 0.1) {
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
                        return;
                    }
                }
            }

            // Try next direction. Make sure not to move backwards
            do {
                d.value = (d.value + 1) % 4;
            } while (d.value == backwards);
        }
    }
}

// Progress enemies along path
void MoveEnemy(flecs::iter& it, 
    flecs::column<transform::Position3> p,
    flecs::column<Direction> d) 
{
    auto ecs = it.world();
    auto g = it.column<const Game>(3);
    const Level* lvl = g->level.get<Level>();

    for (int i = 0; i < it.count(); i ++) {
        find_path(p[i], d[i], lvl);
        p[i].x += dir[d[i].value].x * it.delta_time();
        p[i].z += dir[d[i].value].y * it.delta_time();
    }
}

// Init systems
void init_systems(flecs::world& ecs) {
    ecs.system<>("SpawnEnemy", "Game:Game")
        .interval(1.0)
        .action(SpawnEnemy);

    ecs.system<transform::Position3, Direction>("MoveEnemy", "Game:Game, ANY:Enemy")
        .action(MoveEnemy);
}

int main(int argc, char *argv[]) {
    flecs::world ecs;

    ecs.enable_tracing(1);

    ecs.import<flecs::components::transform>();
    ecs.import<flecs::components::graphics>();
    ecs.import<flecs::components::geometry>();
    ecs.import<flecs::components::gui>();
    ecs.import<flecs::components::physics>();
    ecs.import<flecs::systems::transform>();
    ecs.import<flecs::systems::sdl2>();
    ecs.import<flecs::systems::sokol>();

    auto game = ecs.entity("Game")
        .set<Game>({});

    init_ui(ecs, game);

    init_prefabs(ecs, game);

    init_level(ecs, game);

    init_systems(ecs);

    while(ecs.progress()) { }
}