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
    # Setup stage list
    stages = [(SimpleAIAgent(), simple_wall_map)]
    if difficulty == "hard":
        stages.append((SimpleAIAgent(), cross_wall_map))#(DQNAgent("dqn_snake_hard.pth"), 0.12))

    for stage, (ai, map_func) in enumerate(stages):
        game = SnakeGame(30, 30, max_apples=10, difficulty=difficulty, obstacle_map=map_func)
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

                if mode == "ai":
                    ai_direction = ai.get_action(game.snake2, game.food, game.snake1, game.obstacles)
                    game.update_snake2(ai_direction)

                game.draw(screen, mode)

                if game.is_game_over():
                    elapsed = int(time.time() - start_time)
                    screen.fill((30, 30, 30))

                    if mode == "human":
                        text = f"You win! Time: {elapsed} sec" if game.score1 >= game.max_apples else f"You lose! Time: {elapsed} sec"
                    elif mode == "ai":
                        if game.snake1_dead:
                            text = f"AI wins! You lose in {elapsed} sec"
                        elif game.snake2_dead:
                            text = f"You win! AI dies in {elapsed} sec"
                        elif game.score1 >= game.max_apples:
                            text = f"You win by apples! Time: {elapsed} sec"
                        elif game.score2 >= game.max_apples:
                            text = f"AI wins by apples! Time: {elapsed} sec"
                        else:
                            text = f"Draw! Time: {elapsed} sec"

                    label = font.render(text, True, (255, 255, 255))
                    screen.blit(label, (60, 300))
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
