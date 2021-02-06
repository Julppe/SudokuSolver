import cv2
import numpy as np


# Help function for printing the solved board into console.
# The lists are reversed at this phase to match the picture with the solution.
def print_board(sudoku):
    y = 0
    for i in range(0, 9)[::-1]:
        if y % 3 == 0:
            print("-------------------------")
        print("| {} {} {} | {} {} {} | {} {} {} |".format(sudoku[i][8], sudoku[i][7], sudoku[i][6], sudoku[i][5],
                                                          sudoku[i][4], sudoku[i][3], sudoku[i][2], sudoku[i][1],
                                                          sudoku[i][0]))
        y += 1
    print("-------------------------")


# Locates the current square into a 3 x 3 square.
def find_big_square(i, j):
    return (i // 3) * 3, (j // 3) * 3


# Make a one dimensional list of a 3 x 3 square.
def make_big_square(board, i, j):
    big_square = board[i][j:j + 3] + board[i + 1][j:j + 3] + board[i + 2][j:j + 3]
    return big_square


# Find the next empty square to fill.
def find_empty(board):

    i = 0
    j = 0

    while i < 9:
        j = 0
        while j < 9:
            if board[i][j] == 0:
                return [i, j]
            else:
                j += 1
        i += 1

    return [9, 9]


# Find a number that fits the current square.
def find_number(i, j, board):

    row = board[i]

    column = []
    for x in range(0, 9):
        column.append(board[x][j])

    pos = find_big_square(i, j)
    pos1, pos2 = pos
    big_square = make_big_square(board, pos1, pos2)

    for a in range(1, 10):
        if a not in row and a not in column and a not in big_square:
            board[i][j] = a

            if solve(board)[0]:
                return a

            board[i][j] = 0

    return 0


# Solves a 9 x 9 sudoku grid using the former functions.
def solve(sudoku):
    empty = find_empty(sudoku)
    i = empty[0]
    j = empty[1]
    if empty == [9, 9]:
        return True, sudoku

    number = find_number(i, j, sudoku)

    if number != 0:
        sudoku[i][j] = number

        if solve(sudoku)[0]:
            return True, sudoku

        sudoku[i][j] = 0

    return False, sudoku


# Trains the model to recognise numbers in the picture of a sudoku.
# In the data empty squares are recognised as 0's to minimize later transformations.
def train_model():
    samples = np.loadtxt('samples.data', np.float32)
    responses = np.loadtxt('responses.data', np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    return model


# Creates a 9 x 9 two dimensional list from an image of a sudoku.
def img_to_matrix(model):
    matrix = [[0 for i in range(9)] for j in range(9)]

    orig_image = cv2.imread(input("Input the name of the image file:"))
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(image, 255, 1, 1, 11, 2)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    for area in contours:
        if cv2.contourArea(area) > 50:
            [x, y, width, height] = cv2.boundingRect(area)
            if 200 > height > 20:
                roi = image[y:y + height, x:x + width]
                roi = np.float32(cv2.resize(roi, (20, 20)).reshape((1, 400)))
                val, result, neighb, dist = model.findNearest(roi, k=1)
                num = int(result[0][0])
                matrix[i // 9][i % 9] = num
                i += 1

    return matrix, orig_image


# Main function to operate the solver.
def main():
    model = train_model()
    board, orig_image = img_to_matrix(model)

    sudoku = solve(board)
    print_board(sudoku[1])

    cv2.imshow('original', orig_image)
    cv2.waitKey(0)


main()
