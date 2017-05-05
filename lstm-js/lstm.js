var la = linearAlgebra(),
    Matrix = la.Matrix,
    Vector = la.Vector;

var lstm = lstmInit(Matrix);

var S1 = Vector.zero(lstm.num_cells * 6);
var S2 = Vector.zero(lstm.num_cells * 6);
var S3 = Vector.zero(lstm.num_cells * 6);

function toOneHot(c) {
    var v = la.Vector.zero(lstm.num_input);
    v.data[lstm.encode(c)] = 1;
    return v;
}

function fromOneHot(a) {
    var max = 0;
    for (var i = 1; i < a.data.length; ++i) {
        if (a[i] > a[max]) max = i;
    }
    return max;
}

function forwardpass(c) {
    var x = toOneHot(c);

}