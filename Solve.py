#A Backtracking program in Python to solve Sudoku problem

#A Function to print the Grid
def print_grid(arr):
    for i in range(9):
        for j in range(9):
            print(arr[i][j], end=" "),
        print()

def find_empty_location(arr,l):
    for row in range(9):
        for col in range(9):
            if(arr[row][col]==0):
                l[0]=row
                l[1]=col
                return True
        return False
    
def used_in_row(arr,row,num):
    for i in range(9):
        if(arr[row][i]==num):
            return True
    return False

def used_in_col(arr,col,num):
    for i in range(9):
        if(arr[i][col]==num):
            return True
    return False

def used_in_box(arr,row,col,num):
    for i in range(3):
        for j in range(3):
            if(arr[i+row][j+col]==num):
                return True
    return False

def check_if_location_is_safe(arr,row,col,num):
    return (not used_in_row(arr,row,num) and
           (not used_in_col(arr,col,num) and
           (not used_in_box(arr,row-row%3,col-col%3,num))))

def notInRow(arr,row):
    st = set()

    for i in range(0,9):
        if arr[row][i] in st:
            return False
        
        if arr[row][i]!=0:
            st.add(arr[row][i])

    return True

def notInCol(arr,col):
    st = set()

    for i in range(0,9):
        if arr[i][col] in st:
            return False
        
        if arr[i][col]!=0:
            st.add(arr[i][col])

    return True

def notInBox(arr, startRow, startCol):
    st = set()

    for row in range(0,3):
        for col in range(0,3):
            curr = arr[row + startRow][col + startCol]

            if curr in st:
                return False

            if curr != 0:
                st.add(curr)

    return True

def isValid(arr,row,col):
    return (notInRow(arr,row) and notInCol(arr,col) and
            notInBox(arr, row-row%3, col-col%3))

def isValidConfig(arr):
    for i in range(0,9):
        for j in range(0,9):
            if not isValid(arr,i,j):
                return False
            
    return True

def solve_sudoku(arr):
    if not isValidConfig(arr):
        return False

    l=[0,0]

    if not find_empty_location(arr,l):
        return True
    
    row = l[0]
    col = l[1]

    for num in range(1,10):
        if check_if_location_is_safe(arr,row,col,num):
            arr[row][col]=num

            if solve_sudoku(arr):
                return True
            
            arr[row][col]=0
    
    return False

