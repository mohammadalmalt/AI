import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class BackgammonEnvironment:
    def __init__(self):
        self.board = np.zeros(26, dtype=np.int32)
        self.dice = None
        self.player_turn = 1
        self.done = False
        self.move_count = 0
        self.reset()
    
    def reset(self):
        self.board = np.zeros(26, dtype=np.int32)
        self.board[0] = 2    # White
        self.board[5] = -5   # Black
        self.board[7] = -3   # Black
        self.board[11] = 5   # White
        self.board[12] = -5  # Black
        self.board[16] = 3   # White
        self.board[18] = 5   # White
        self.board[23] = -2  # Black
        self.board[24] = 0   # White bar
        self.board[25] = 0   # Black bar
        self.player_turn = 1
        self.done = False
        self.move_count = 0
        self.roll_dice()
        return self.get_state()
    
    def roll_dice(self):
        self.dice = [random.randint(1, 6), random.randint(1, 6)]
        if self.dice[0] == self.dice[1]:
            self.dice = self.dice * 2
        return self.dice
    
    def get_state(self):
        state = self.board.copy() * self.player_turn  # 26 elements
        state_with_player = np.append(state, self.player_turn)  # 27 elements
        dice_array = np.zeros(4)  # Fixed 4 elements
        if self.dice:
            dice_array[:len(self.dice)] = self.dice
        return np.append(state_with_player, dice_array)  # 31 elements
    
    def is_valid_move(self, from_pos, steps):
        direction = self.player_turn
        if from_pos == 24 and direction == 1:  # White re-entering from bar
            to_pos = steps - 1  # Target 0-5
            if self.board[24] <= 0:  # No White pieces on bar
                return False
        elif from_pos == 25 and direction == -1:  # Black re-entering from bar
            to_pos = 23 - (steps - 1)  # Target 23-18
            if self.board[25] >= 0:  # No Black pieces on bar
                return False
        else:  # Regular move
            to_pos = from_pos + direction * steps
            if to_pos < 0 or to_pos > 23:
                if self.can_bear_off():
                    return self.is_valid_bear_off(from_pos, steps)
                return False
            if (direction == 1 and self.board[from_pos] <= 0) or \
               (direction == -1 and self.board[from_pos] >= 0):
                return False
        
        # Check if destination is blocked
        if (direction == 1 and self.board[to_pos] < -1) or \
           (direction == -1 and self.board[to_pos] > 1):
            return False
        return True
    
    def can_bear_off(self):
        if self.player_turn == 1:
            return all(self.board[i] <= 0 for i in range(18)) and self.board[24] == 0
        else:
            return all(self.board[i] >= 0 for i in range(6, 24)) and self.board[25] == 0
    
    def is_valid_bear_off(self, from_pos, steps):
        direction = self.player_turn
        if direction == 1:
            if from_pos + steps > 23 and self.board[from_pos] > 0:
                for i in range(from_pos + 1, 24):
                    if self.board[i] > 0:
                        return from_pos + steps == 24
                return True
        else:
            if from_pos - steps < 0 and self.board[from_pos] < 0:
                for i in range(0, from_pos):
                    if self.board[i] < 0:
                        return from_pos - steps == -1
                return True
        return False
    
    def get_valid_moves(self):
        valid_moves = []
        bar_pos = 24 if self.player_turn == 1 else 25
        if (self.player_turn == 1 and self.board[24] > 0) or \
           (self.player_turn == -1 and self.board[25] < 0):
            for die in self.dice:
                if self.is_valid_move(bar_pos, die):
                    valid_moves.append((bar_pos, die))
            """
            if not valid_moves and self.board[bar_pos] != 0:
                print(f"Player {self.player_turn} stuck on bar with dice {self.dice}, board: {self.board[:24]}")
            """
        else:
            for from_pos in range(24):
                for die in self.dice:
                    if self.is_valid_move(from_pos, die):
                        valid_moves.append((from_pos, die))
        return valid_moves
    
    def execute_move(self, from_pos, steps):
        direction = self.player_turn
        if from_pos == 24 and direction == 1:  # White from bar
            to_pos = steps - 1
        elif from_pos == 25 and direction == -1:  # Black from bar
            to_pos = 23 - (steps - 1)
        else:
            to_pos = from_pos + direction * steps
        self.move_count += 1
        
        if (direction == 1 and self.board[from_pos] <= 0) or \
           (direction == -1 and self.board[from_pos] >= 0):
            print(f"Move {self.move_count}: Invalid move attempt from {from_pos}")
            return False
        
        if (direction == 1 and to_pos > 23) or (direction == -1 and to_pos < 0):
            self.board[from_pos] -= direction
            self.check_game_over()
            return True
        
        if self.board[to_pos] == -direction:  # Hit opponent
            self.board[to_pos] = 0
            bar_pos = 25 if direction == 1 else 24
            self.board[bar_pos] -= direction
        
        self.board[from_pos] -= direction
        self.board[to_pos] += direction
        
        assert steps in self.dice, f"Invalid die {steps} not in {self.dice}"
        self.dice.remove(steps)
        self.check_game_over()
        return True
    
    def check_game_over(self):
        white_pieces = sum(max(0, self.board[i]) for i in range(24)) + self.board[24]
        black_pieces = sum(min(0, self.board[i]) for i in range(24)) - self.board[25]
        if white_pieces == 0 or black_pieces == 0:
            self.done = True
            print(f"Move {self.move_count}: Game over - White: {white_pieces}, Black: {black_pieces}")
        return self.done
    
    def get_winner(self):
        if not self.done:
            return 0
        white_pieces = sum(max(0, self.board[i]) for i in range(24)) + self.board[24]
        return 1 if white_pieces == 0 else -1
    
    def next_turn(self):
        self.player_turn *= -1
        self.roll_dice()
    
    def step(self, action):
        from_pos, steps = action
        if not self.is_valid_move(from_pos, steps):
            return self.get_state(), -10, self.done, {}
        
        direction = self.player_turn
        to_pos = from_pos if from_pos >= 24 else (steps - 1 if from_pos == 24 else 23 - (steps - 1) if from_pos == 25 else from_pos + direction * steps)
        hit = self.board[to_pos] == -direction if to_pos in range(24) else False
        
        success = self.execute_move(from_pos, steps)
        if not success:
            return self.get_state(), -10, self.done, {}
        
        if not self.dice:
            self.next_turn()
        
        reward = 0.01  # Base reward for any move
        if success:
            if (direction == 1 and to_pos > 23) or (direction == -1 and to_pos < 0):
                reward += 0.5  # Bearing off bonus
            if hit:
                reward += 0.5  # Increased hit reward (was 0.2)
        
        # Penalty for pieces on bar
        bar_pieces = self.board[24] if direction == 1 else -self.board[25]
        reward -= 0.1 * bar_pieces  # -0.1 per piece on bar
        
        if self.done:
            winner = self.get_winner()
            reward += 10 if winner == direction else -10
        
        # Add move limit to prevent infinite games
        if self.move_count >= 500:
            self.done = True
            reward -= 5  # Penalty for draw
            print(f"Move {self.move_count}: Game ended in draw due to move limit")
        
        return self.get_state(), reward, self.done, {}

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Increased from 0.01 for more exploration
        self.epsilon_decay = 0.99  # Slowed from 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
    
    def _build_model(self):
        class DQN(nn.Module):
            def __init__(self, input_size, output_size):
                super(DQN, self).__init__()
                self.fc1 = nn.Linear(input_size, 256)
                self.fc2 = nn.Linear(256, 256)
                self.fc3 = nn.Linear(256, 128)
                self.fc4 = nn.Linear(128, output_size)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.relu(self.fc3(x))
                x = self.fc4(x)
                return x
        return DQN(self.state_size, self.action_size)
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_moves):
        if not valid_moves:
            return None
        valid_action_indices = [self.move_to_action_index(move) for move in valid_moves]
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state_tensor)[0]
        valid_q_values = {idx: act_values[idx].item() for idx in valid_action_indices}
        best_action_idx = max(valid_q_values, key=valid_q_values.get)
        return self.action_index_to_move(best_action_idx)
    
    def move_to_action_index(self, move):
        from_pos, steps = move
        return from_pos * 6 + steps - 1
    
    def action_index_to_move(self, action_idx):
        from_pos = action_idx // 6
        steps = (action_idx % 6) + 1
        return (from_pos, steps)
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([m[0] for m in minibatch]).to(self.device)
        actions = [self.move_to_action_index(m[1]) for m in minibatch]
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        next_states = torch.FloatTensor([m[3] for m in minibatch]).to(self.device)
        dones = torch.FloatTensor([m[4] for m in minibatch]).to(self.device)
        q_values = self.model(states)
        next_q_values = self.target_model(next_states).detach()
        targets = q_values.clone()
        for i in range(batch_size):
            action_idx = actions[i]
            targets[i][action_idx] = rewards[i] if dones[i] else rewards[i] + self.gamma * torch.max(next_q_values[i])
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(q_values, targets)
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_self_play(iterations=100, games_per_iteration=20, training_steps=100):
    env = BackgammonEnvironment()
    state_size = 26 + 1 + 4
    action_size = 26 * 6
    agent1 = DQNAgent(state_size, action_size)  # White
    agent2 = DQNAgent(state_size, action_size)  # Black
    best_win_rate_white = 0.0
    
    for iteration in range(iterations):
        print(f"\n=== Iteration {iteration+1}/{iterations} ===")
        print(f"Starting Epsilon: Agent1={agent1.epsilon:.2f}, Agent2={agent2.epsilon:.2f}")
        memory_buffer = deque(maxlen=10000)
        total_moves = 0
        
        # Reset epsilon periodically to encourage exploration
        if iteration % 50 == 0 and iteration > 0:
            agent1.epsilon = max(0.1, agent1.epsilon)
            agent2.epsilon = max(0.1, agent2.epsilon)
            print(f"Reset epsilon to {agent1.epsilon:.2f} for exploration")
        
        # Self-play games
        for game in range(games_per_iteration):
            state = env.reset()
            print(f"  Game {game+1}/{games_per_iteration} started")
            while not env.done:
                valid_moves = env.get_valid_moves()
                if not valid_moves:
                    env.next_turn()
                    state = env.get_state()
                    continue
                
                current_agent = agent1 if env.player_turn == 1 else agent2
                action = current_agent.act(state, valid_moves)
                if action is None:
                    env.next_turn()
                    state = env.get_state()
                    continue
                
                next_state, reward, done, _ = env.step(action)
                memory_buffer.append((state, action, reward * env.player_turn, next_state, done))
                state = next_state
            
            winner = env.get_winner()
            total_moves += env.move_count
            print(f"  Game {game+1} finished, Moves: {env.move_count}, Winner: {winner}")
            for i in range(len(memory_buffer) - 1, -1, -1):
                s, a, r, ns, d = memory_buffer[i]
                if d:
                    memory_buffer[i] = (s, a, winner * env.player_turn, ns, d)
                    break
        
        # Training steps
        print(f"Iteration {iteration+1}: Starting training, Buffer size: {len(memory_buffer)}")
        for step in range(training_steps):
            if len(memory_buffer) < 64:
                break
            minibatch = random.sample(memory_buffer, 64)
            for state, action, reward, next_state, done in minibatch:
                player_turn = state[26]
                agent1.remember(state, action, reward if player_turn == 1 else -reward, next_state, done)
                inverted_state = state.copy()
                inverted_state[:26] *= -1
                inverted_state[26] *= -1
                agent2.remember(inverted_state, action, -reward if player_turn == 1 else reward, next_state, done)
            agent1.replay(64)
            agent2.replay(64)
            if step % 50 == 0:
                print(f"  Training step {step}/{training_steps}, "
                      f"Epsilon: Agent1={agent1.epsilon:.2f}, Agent2={agent2.epsilon:.2f}")
        
        # Evaluation and progress
        avg_moves = total_moves / games_per_iteration
        win_rate_white = evaluate_agent(agent1, player=1, games=10)  # White
        win_rate_black = evaluate_agent(agent2, player=-1, games=10)  # Black
        print(f"Iteration {iteration+1} Summary:")
        print(f"  Learning Rate: {agent1.learning_rate}")
        print(f"  Ending Epsilon: Agent1={agent1.epsilon:.2f}, Agent2={agent2.epsilon:.2f}")
        print(f"  Average Moves per Game: {avg_moves:.1f}")
        print(f"  Win Rate (White): {win_rate_white:.2f}")
        print(f"  Win Rate (Black): {win_rate_black:.2f}")
        
        if win_rate_white > best_win_rate_white:
            best_win_rate_white = win_rate_white
            torch.save(agent1.model.state_dict(), "best_backgammon_model_white.pt")
            print(f"  New best model for White saved with win rate: {best_win_rate_white:.2f}")
        
        agent1.update_target_model()
        agent2.update_target_model()
    
    return agent1, agent2

def evaluate_agent(agent, player, games=10):
    env = BackgammonEnvironment()
    wins = 0
    for _ in range(games):
        state = env.reset()
        while not env.done:
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                env.next_turn()
                continue
            old_epsilon = agent.epsilon
            agent.epsilon = 0
            action = agent.act(state, valid_moves)
            agent.epsilon = old_epsilon
            if action is None:
                env.next_turn()
                continue
            state, _, done, _ = env.step(action)
        if env.get_winner() == player:  # Check if this agent’s player won
            wins += 1
    win_rate = wins / games
    print(f"Evaluation - Win rate for Player {player}: {win_rate:.2f}")
    return win_rate

def play_against_human(agent_black):
    env = BackgammonEnvironment()
    state = env.reset()
    
    print("Welcome to Backgammon! You are White (Player 1), AI is Black (Player -1)")
    print("Board positions: 0-23 (board), 24 (White bar), 25 (Black bar)")
    print("Positive numbers = White pieces, Negative = Black pieces")
    print("Enter moves as 'from_pos steps' (e.g., '11 5' to move from 11 with 5)")
    
    game_over = False
    while not game_over:
        print("\nCurrent board:")
        print(env.board.reshape(2, 13))
        print(f"Player {env.player_turn}'s turn, Dice: {env.dice}")
        
        if env.player_turn == 1:  # Human (White)
            valid_moves = env.get_valid_moves()
            print(f"Valid moves: {valid_moves}")
            if not valid_moves:
                print("No valid moves, switching turns")
                env.next_turn()
                continue
            
            while True:
                move_input = input("Your move (or 'quit' to exit): ").strip()
                if move_input.lower() == 'quit':
                    print("Game ended by player")
                    return
                try:
                    from_pos, steps = map(int, move_input.split())
                    move = (from_pos, steps)
                    if move in valid_moves:
                        break
                    else:
                        print("Invalid move! Choose from valid moves.")
                except ValueError:
                    print("Invalid input! Use format 'from_pos steps' (e.g., '11 5')")
            
            state, reward, done, _ = env.step(move)
            print(f"You moved: {move}")
        
        else:  # AI (Black)
            valid_moves = env.get_valid_moves()
            print(f"Valid moves for AI: {valid_moves}")
            if not valid_moves:
                print("No valid moves for AI, switching turns")
                env.next_turn()
                continue
            
            old_epsilon = agent_black.epsilon
            agent_black.epsilon = 0
            action = agent_black.act(state, valid_moves)
            agent_black.epsilon = old_epsilon
            state, reward, done, _ = env.step(action)
            print(f"AI chose move: {action}")
        
        game_over = done
    
    winner = env.get_winner()
    print("\nFinal board:")
    print(env.board.reshape(2, 13))
    print(f"Game over! Winner: {'White' if winner == 1 else 'Black' if winner == -1 else 'None'}")

if __name__ == "__main__":
    print("Starting self-play training for backgammon...")
    agent1, agent2 = train_self_play(iterations=500, games_per_iteration=20, training_steps=100)
    print("Training complete!")
    
    try:
        agent2.model.load_state_dict(torch.load("best_backgammon_model_white.pt"))
        print("Loaded best model for Black (Agent2)")
    except FileNotFoundError:
        print("No saved model found, using trained Agent2")
    
    play_against_human(agent2)