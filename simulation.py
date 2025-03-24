import cv2
import numpy as np

def draw_classroom(letter):
    img = np.ones((500, 800, 3), dtype=np.uint8) * 255  # White background
    
    # Stick figure key points
    head_center = (150, 150)
    body_start = (150, 200)
    body_end = (150, 300)
    left_leg = (120, 400)
    right_leg = (180, 400)
    left_arm = (120, 250)
    right_arm_start = (200, 250)  # Fixed right arm position
    
    cv2.circle(img, head_center, 30, (0, 0, 0), 2)  # Head
    cv2.line(img, body_start, body_end, (0, 0, 0), 3)  # Body
    cv2.line(img, body_end, left_leg, (0, 0, 0), 3)  # Left leg
    cv2.line(img, body_end, right_leg, (0, 0, 0), 3)  # Right leg
    cv2.line(img, body_start, left_arm, (0, 0, 0), 3)  # Left arm
    cv2.line(img, body_start, right_arm_start, (0, 0, 0), 4)  # Right arm
    
    # Board
    board_top_left = (400, 100)
    board_bottom_right = (700, 250)
    cv2.rectangle(img, board_top_left, board_bottom_right, (255, 0, 0), 3)
    
    # Letters A, B, C
    letter_positions = {"A": (450, 180), "B": (530, 180), "C": (610, 180)}
    for key, pos in letter_positions.items():
        cv2.putText(img, key, pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Pointer stick
    pointer_end = letter_positions.get(letter, (450, 180))
    cv2.line(img, right_arm_start, pointer_end, (0, 0, 255), 3)  # Stick
    
    return img

if __name__ == "__main__":
    while True:
        try:
            with open("detected_letter.txt", "r") as file:
                detected_letter = file.read().strip()
        except FileNotFoundError:
            detected_letter = "A"  # Default to A if no file exists

        classroom_img = draw_classroom(detected_letter)
        cv2.imshow("Classroom Simulation", classroom_img)
        if cv2.waitKey(500) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cv2.destroyAllWindows()
