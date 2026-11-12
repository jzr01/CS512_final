import numpy as np


class Renderer2D:
    def __init__(self, mode, size=720):
        import pygame
        self.pg, self.mode, self.size = pygame, mode, size
        pygame.display.init()
        pygame.font.init()
        self.surface = pygame.display.set_mode((size, size)) if mode == "human" else pygame.Surface((size, size))
        self.clock = pygame.time.Clock()

    def render(self, env):
        pg, surface = self.pg, self.surface
        surface.fill((19, 25, 39))
        scale = self.size / env.map_size
        def point(p):
            return int(p[0] * scale), int(self.size - p[1] * scale)
        pg.draw.circle(surface, (35, 105, 65), point(env.target), int(env.goal_radius * scale))
        for center, radius in env.obstacles:
            pg.draw.circle(surface, (125, 132, 148), point(center), int(radius * scale))
        for i, agent in enumerate(env.agents):
            for j in agent.cache.entries:
                if j > i and agent.detector.status(j) not in ("UNAVAILABLE", "SUSPECTED"):
                    pg.draw.line(surface, (52, 74, 103), point(env.positions[i]), point(env.positions[j]), 1)
        for i, pos in enumerate(env.positions):
            if len(env.trajectories[i]) > 1:
                pg.draw.lines(surface, (70, 95, 125), False, [point(p) for p in env.trajectories[i]], 1)
            suspected = any(a.detector.status(i) in ("SUSPECTED", "UNAVAILABLE") for a in env.agents if a.robot_id != i)
            color = (210, 65, 75) if env.crashed[i] else (230, 180, 60) if suspected else (65, 175, 235)
            pg.draw.circle(surface, color, point(pos), max(4, int(env.robot_radius * scale)))
            font = pg.font.Font(None, 20)
            surface.blit(font.render(str(i), True, (255,255,255)), point(pos))
        if self.mode == "human":
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.close()
                    return
            pg.display.flip()
            self.clock.tick(env.metadata["render_fps"])
        return np.transpose(pg.surfarray.array3d(surface), (1, 0, 2))

    def close(self):
        self.pg.quit()


if __name__ == "__main__":
    from distributed_swarm.evaluation.evaluate import main
    main(default_kind="2d")
