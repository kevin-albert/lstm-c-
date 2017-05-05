var V32 = typeof Float32Array == 'function' ? Float32Array : Array;

function rows(M) {
    return M.length;
}

function cols(M) {
    return typeof M[0] == 'object' ? M[0].length : M.length;
}

function dot(a, b) {
    var a_ = typeof a[0] == 'object' ? a[0] : a;
    var b_ = typeof b[0] == 'object' ? b[0] : b;
    if (a_.length != b_.length) {
        throw new Error('invalid dot product: (' + a_.length + ' * ' + b_.length + ')');
    }
    var sum = 0;
    for (var i = 0; i < a_.length; ++i)
        sum += a_[i] * b_[i];
    return sum;
}

function matrix(rows, cols) {
    var M = new Array(rows);
    for (var i = 0; i < rows; ++i) M[i] = new V32(cols);
    return M;
}

function multiply(A, B) {

    if (cols(A) != rows(B)) throw new Error('invalid matrix product: (' + rows(A) + 'x' + cols(A) + ' * ' + rows(B) + 'x' + cols(B));
    var C = matrix(rows(B), rows(A));
    for (var i = 0; i < rows(C); ++i) {
        for (var i = 0; i < rows(A); ++i) {

        }
    }
    return C;
}