class Matrix(object):
    """Class for matrices, expressed as lists of lists
    """

    def __init__(self, data):
        """Creates a matrix from a list of lists of data
        Constructor: Complex(list(list(float)))
        """
        self.columns = len(data[0])
        self.rows = len(data)
        self.data = []
        for line in data:
            assert len(line) == self.columns, "not rectangular data"
            self.data.append(line)

    def __add__(self, other):
        """Adds two matrices together
        Matrix + Matrix -> Matrix
        """
        assert self.rows == other.rows, "Matrices not same height"
        assert self.columns == other.columns, "Matrices not same width"
        return Matrix([[self.data[i][j]+other.data[i][j]for j in range(self.columns)]for i in range(self.rows)])

    def __mul__(self, other):
        """Multiply a matrix by another matrix, or by a scalar
        Matrix*Matrix -> Matrix
        Matrix*float -> Matrix
        """
        if type(other) == int or type(other) == float:
            return Matrix([[other*self.data[i][j]
                            for j in range(self.columns)]for i in range(self.rows)])
        assert self.columns == other.rows, "Can't multiply these matrices"
        return Matrix([[sum([self.data[i][k]*other.data[k][j]for k in range(self.columns)])
                        for j in range(other.columns)]for i in range(self.rows)])

    def __rmul__(self, other):
        """Multiply a scalar by a Matrix
        float*Matrix -> Matrix
        """
        return self*other

    def __str__(self):
        """Provides a readable representation of matrix data
        """
        returnstring = ""
        for line in self.data:
            returnstring += str(line)
            returnstring += "\n"
        return returnstring

    def __sub__(self, other):
        """Subtracts one matrix from another
        Matrix - Matrix -> Matrix
        """
        assert self.rows == other.rows, "Matrices not same height"
        assert self.columns == other.columns, "Matrices not same width"
        return Matrix([[self.data[i][j]-other.data[i][j]for j in range(self.columns)]for i in range(self.rows)])

    def det(self):
        """Returns the determinant of a square matrix using row operations
        Matrix.det() => float
        """
        assert self.columns == self.rows, "not square matrix"
        if self.columns == 2:
            return self.data[0][0]*self.data[1][1]-self.data[0][1]*self.data[1][0]
        detmatrix = Matrix(self.data)
        if detmatrix.data[0][0]:
            mult = detmatrix.data[0][0]
            detmatrix = detmatrix.rowscale(0, 1/detmatrix.data[0][0])
            for row in range(1, self.rows):
                detmatrix = detmatrix.rowadd(0, -1*detmatrix.data[row][0], row, 1)
            return Matrix([detmatrix.data[i][1:]for i in range(1, self.rows)]).det()*mult
        for element in range(self.columns):
            if self.data[element][0] != 0:
                return detmatrix.rowswap(0,element).det()
        else:
            return 0

    def inverse(self):
        """returns the inverse of the matrix using row operations
        Matrix.inverse() => Matrix
        """
        assert self.det() != 0, "Non-invertible matrix"
        augmatrix = Matrix([([self.data[i][j]for j in range(self.columns)] +
                             [1 if k == i else 0 for k in range(self.columns)])for i in range(self.rows)])
        # augments matrix with the identity
        for column in range(self.columns):
            for row in range(column,self.rows):
                if augmatrix.data[row][column]:
                    augmatrix = augmatrix.rowswap(row, column)
                    break
            augmatrix = augmatrix.rowscale(column, 1/augmatrix.data[column][column])
            for row in range(column+1, self.rows):
                augmatrix = augmatrix.rowadd(column, -1*augmatrix.data[row][column], row, 1)
        for column in range(self.columns-1,  -1, -1):
            for row in range(column-1, -1, -1):
                augmatrix = augmatrix.rowadd(column, -1*augmatrix.data[row][column], row, 1)
        return Matrix([augmatrix.data[i][self.rows:]for i in range(self.rows)])

    def rowscale(self, row, scalar):
        """Replaces a row in the matrix by a multiple of itself
        Matrix.rowscale(int, float) => Matrix
        """
        return Matrix([[scalar*self.data[i][j] if i == row else self.data[i][j] for j in range(self.columns)]
                       for i in range(self.rows)])

    def rowadd(self, rowa, ascale, rowb, bscale):
        """Adds ascale*rowa and adds it to bscale*rowb in rowb
        Matrix.rowscale(int, float, int, float) => Matrix
        """
        return Matrix([[bscale*self.data[i][j]+ascale*self.data[rowa][j] if i == rowb else self.data[i][j]
                        for j in range(self.columns)]for i in range(self.rows)])

    def rowswap(self, rowa, rowb):
        """Replaces the contents of rowa with row b and vice versa
        Matrix.rowswap(int, int) => Matrix
        """
        returnmatrix = []
        for row in range(len(self.data)):
            if row == rowa:
                returnmatrix.append(self.data[rowb])
            elif row == rowb:
                returnmatrix.append(self.data[rowa])
            else:
                returnmatrix.append(self.data[row])
        return Matrix(returnmatrix)

    def transpose(self):
        ans = list()
        A = self.data
        l = len(A[0]) if isinstance(A[0], list) else 1
        shX = len(A)
        shY = l
        for y in range(shY):
            ans.append(list())
            for x in range(shX):
                ans[y].append(A[x][y])
        return Matrix(ans)


def ones(n):
    ans = list()
    for i in range(n):
        ans.append(list())
        for j in range(n):
            l = 1.0 if i == j else 0.0
            ans[i].append(l)
    return ans


# [[a],[b],[c]] <= [a, b, c]
def make_matrix(A):
    ans = list()
    for i, it in enumerate(A):
        ans.append(list())
        ans[i].append(it)
    return ans


# [[a],[b],[c]] => [a, b, c]
def make_vector(A):
    ans = list()
    for i, it in enumerate(A):
        ans.append(A[i][0])
    return ans