"""
AracSim_RL — Standalone Demo
============================

Loads trained checkpoints (arac_latest.pt + tc_latest.pt) and runs an
interactive visualization where the Traffic Monster (TC) chooses road
parameters and the trained PPO agent drives.

Usage:
    python play.py              # default: TC controls difficulty
    python play.py --easy       # fixed easy preset
    python play.py --hard       # fixed hard preset
    python play.py --episodes N # quit after N episodes (default: unlimited)

Keyboard during play:
    R     restart current scenario
    N     new random scenario
    ESC   quit
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pygame

import config
from agents.arac_ppo       import AracPPO
from agents.trafik_canavari import TrafıkCanavarı, aksiyon_to_params
from data.models           import TCRoadParams
from env.road_env          import RLEnvironment
from render.renderer       import Renderer


CHECKPOINT_DIR = "checkpoints"
ARAC_PT        = os.path.join(CHECKPOINT_DIR, "arac_latest.pt")
TC_PT          = os.path.join(CHECKPOINT_DIR, "tc_latest.pt")


PRESETS = {
    "easy":    TCRoadParams(curve_amplitude=0.18, grade_max=0.02,
                            width_min=9.0, friction_min=0.90),
    "default": TCRoadParams(curve_amplitude=0.35, grade_max=0.05,
                            width_min=7.5, friction_min=0.55),
    "hard":    TCRoadParams(curve_amplitude=0.65, grade_max=0.08,
                            width_min=5.5, friction_min=0.20),
}


def _check_checkpoints() -> None:
    missing = [p for p in (ARAC_PT, TC_PT) if not os.path.isfile(p)]
    if missing:
        print("ERROR: checkpoint(s) missing:")
        for p in missing:
            print(f"  - {p}")
        print("\nPlease place arac_latest.pt and tc_latest.pt under "
              f"{CHECKPOINT_DIR}/ — see README.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="AracSim_RL demo player")
    parser.add_argument("--easy",     action="store_true")
    parser.add_argument("--hard",     action="store_true")
    parser.add_argument("--default",  action="store_true",
                        help="fixed medium preset (skip TC ajan)")
    parser.add_argument("--episodes", type=int, default=0,
                        help="N>0: stop after N episodes")
    args = parser.parse_args()

    use_tc = not (args.easy or args.hard or args.default)
    fixed_preset = (
        PRESETS["easy"]    if args.easy    else
        PRESETS["hard"]    if args.hard    else
        PRESETS["default"] if args.default else None
    )

    _check_checkpoints()

    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    pygame.display.set_caption("AracSim_RL — Demo")
    font = pygame.font.SysFont("consolas", 22)

    env  = RLEnvironment()
    arac = AracPPO()
    arac.yukle(ARAC_PT)
    print(f"  loaded: {ARAC_PT}")

    tc = TrafıkCanavarı() if use_tc else None
    if use_tc:
        tc.yukle(TC_PT)
        print(f"  loaded: {TC_PT}")
        print("Mode: TC ajan zorluğu seçiyor")
    else:
        print(f"Mode: fixed preset")

    def next_params() -> TCRoadParams:
        if fixed_preset is not None:
            return fixed_preset
        tc_obs = env.get_tc_observation()
        return aksiyon_to_params(tc.aksiyon_sec(tc_obs))

    print("Tuşlar: R = restart | N = yeni yol | ESC = çıkış\n")

    renderer = Renderer(font)
    tc_params = next_params()
    obs = env.reset(tc_params)

    env.mod = "DEMO"
    env._recent_success.clear()
    bolum = 0
    basari = 0
    son_sonuc = "-"
    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(config.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs = env.reset(tc_params)
                elif event.key == pygame.K_n:
                    tc_params = next_params()
                    obs = env.reset(tc_params)

        action, _, _ = arac.aksiyon_sec(obs, deterministik=True)
        obs, _, done, info = env.step(action)
        if done:
            bolum += 1
            ok = bool(info.get("success"))
            if ok:
                basari += 1
                son_sonuc = "GOAL"
            elif info.get("crashed"):
                son_sonuc = "CRASH"
            elif info.get("stuck"):
                son_sonuc = "STUCK"
            else:
                son_sonuc = "TIME"
            env._recent_success.append(1.0 if ok else 0.0)
            if len(env._recent_success) > 20:
                env._recent_success.pop(0)
            env.jenerasyon = bolum
            env.mod = f"DEMO [{son_sonuc}]"
            if args.episodes and bolum >= args.episodes:
                running = False
            else:
                tc_params = next_params()
                obs = env.reset(tc_params)

        renderer.render(screen, env)
        pygame.display.flip()

    pygame.quit()
    if bolum:
        print(f"\nÖzet: {basari}/{bolum} bölüm başarılı "
              f"({basari/bolum*100:.0f}%)")


if __name__ == "__main__":
    main()
