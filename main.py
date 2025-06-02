import pygame
import sys
import time
from game_env import SnakeGame
from agent import HumanAgent, SimpleAIAgent, DQNAgent
from maps import simple_wall_map, cross_wall_map

def choose_difficulty(screen, font):
    options = [
        {"label": "Easy", "value": "easy", "rect": pygame.Rect(100, 120, 200, 50)},
        {"label": "Hard", "value": "hard", "rect": pygame.Rect(100, 200, 200, 50)}
    ]
    return select_menu(screen, font, "Select Difficulty", options)

def choose_mode(screen, font):
    options = [
        {"label": "Play Yourself", "value": "human", "rect": pygame.Rect(100, 120, 200, 50)},
        {"label": "Play vs AI", "value": "ai", "rect": pygame.Rect(100, 200, 200, 50)}
    ]
    return select_menu(screen, font, "Select Mode", options)

def select_menu(screen, font, title, buttons):
    screen_width, screen_height = screen.get_size()

    bg_color = (10, 10, 40)
    button_color = (70, 130, 180)
    hover_color = (100, 160, 210)
    text_color = (255, 255, 255)
    title_font = pygame.font.SysFont("consolas", 48)
    option_font = pygame.font.SysFont("consolas", 32)

    # 按鈕設定
    button_width = 300
    button_height = 60
    button_margin = 30
    total_height = len(buttons) * (button_height + button_margin) - button_margin

    # 動態生成 rects 並置中
    for i, btn in enumerate(buttons):
        x = (screen_width - button_width) // 2
        y = screen_height // 2 - total_height // 2 + i * (button_height + button_margin)
        btn["rect"] = pygame.Rect(x, y, button_width, button_height)

    while True:
        screen.fill(bg_color)
        mouse_pos = pygame.mouse.get_pos()
        click = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click = True

        # 畫標題
        title_render = title_font.render(title, True, (255, 255, 100))
        screen.blit(title_render, (screen_width // 2 - title_render.get_width() // 2, 60))

        # 畫按鈕
        for btn in buttons:
            rect = btn["rect"]
            color = hover_color if rect.collidepoint(mouse_pos) else button_color
            pygame.draw.rect(screen, color, rect, border_radius=15)

            label = option_font.render(btn["label"], True, text_color)
            text_rect = label.get_rect(center=rect.center)
            screen.blit(label, text_rect)

            if click and rect.collidepoint(mouse_pos):
                return btn["value"]

        pygame.display.flip()



def wait_for_key_and_restart(screen, font):
    pygame.event.clear()  # 清除舊事件佇列，避免干擾下一輪
    screen.fill((30, 30, 30))
    msg = font.render("Press any key to return to main menu", True, (255, 255, 255))
    screen.blit(msg, (80, 330))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                waiting = False

def run_game(screen, font, difficulty, mode):
    # 根據模式與難度設計關卡
    if mode == "human":
        if difficulty == "easy":
            stages = [(None, simple_wall_map)]
        else:  # hard
            stages = [(None, simple_wall_map), (None, cross_wall_map)]
    else:  # mode == "ai"
        if difficulty == "easy":
            stages = [(SimpleAIAgent(), simple_wall_map)]
        else:  # hard
            stages = [
                (DQNAgent("dqn_snake_hard.pth"), simple_wall_map),
                (DQNAgent("dqn_snake_hard.pth"), cross_wall_map)
            ]

    for stage, (ai, map_func) in enumerate(stages):
        pygame.event.clear()  # 避免事件殘留造成提前結束
        game = SnakeGame(30, 30, max_apples=1, difficulty=difficulty, obstacle_map=map_func)
        human = HumanAgent()
        paused = False
        start_time = time.time()

        while True:
            screen.fill((0, 0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    human.update_direction(event)

            if not paused:
                direction = human.get_action()
                game.update_snake1(direction)

                if mode == "ai" and ai is not None:
                    ai_direction = ai.get_action(game.snake2, game.food, game.snake1, game.obstacles)
                    game.update_snake2(ai_direction)

                game.draw(screen, mode)

                if game.is_game_over():
                    elapsed = int(time.time() - start_time)
                    screen.fill((30, 30, 30))

                    # 顯示關卡與勝負
                    ai_name = type(ai).__name__.replace("Agent", "") if ai else ""
                    ai_info = f" vs {ai_name}" if ai_name else ""
                    stage_info = f"Stage {stage+1}/{len(stages)}{ai_info} | Time: {elapsed} sec"

                    if mode == "human":
                        result_text = "You win!" if game.score1 >= game.max_apples else "You lose!"
                    elif mode == "ai":
                        if game.snake1_dead:
                            result_text = "AI wins! You died"
                        elif game.snake2_dead:
                            result_text = "You win! AI died"
                        elif game.score1 >= game.max_apples:
                            result_text = "You win by apples!"
                        elif game.score2 >= game.max_apples:
                            result_text = "AI wins by apples!"
                        else:
                            result_text = "Draw!"

                    label1 = font.render(stage_info, True, (255, 255, 255))
                    label2 = font.render(result_text, True, (255, 255, 100))
                    screen.blit(label1, (60, 270))
                    screen.blit(label2, (60, 310))
                    pygame.display.flip()
                    pygame.time.wait(3000)
                    break

            pygame.display.flip()
            pygame.time.Clock().tick(20)


    wait_for_key_and_restart(screen, font)

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((660, 700))
    pygame.display.set_caption("Snake Game")
    font = pygame.font.SysFont("consolas", 24)

    while True:
        difficulty = choose_difficulty(screen, font)
        mode = choose_mode(screen, font)
        run_game(screen, font, difficulty, mode)
