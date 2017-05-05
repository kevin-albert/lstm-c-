var la = linearAlgebra(),
    Matrix = la.Matrix,
    Vector = la.Vector;

var lstm = lstmInit(Matrix);
var S1 = Vector.zero(lstm.num_cells * 6);
var S2 = Vector.zero(lstm.num_cells * 6);
var S3 = Vector.zero(lstm.num_cells * 6);


function sample(n, t) {

    var c = '^';
    var output = '';
    while (n > 0) {
        c = forwardpass(c, t);
        console.log(c);
        output += c;
        --n;
    }

    return output;

    function toOneHot(c) {
        var v = Vector.zero(lstm.num_input);
        v.data[0][lstm.encode(c)] = 1;
        return v.trans();
    }

    function fromOneHot(a) {
        var max = 0;
        var data = a.trans().toArray();
        for (var i = 1; i < data.length; ++i) {
            if (data[i] > data[max]) max = i;
        }
        return lstm.decode(max);
    }

    function fromDist(p) {
        var rand = Math.random();
        var a = p.trans().toArray();
        for (var i = 0; i < a; ++i) {
            rand -= a[i];
            if (rand <= 0) {
                return lstm.decode(i);
            }
        }
        return lstm.decode(a.length);
    }

    function forwardpass(c, t) {
        var x = toOneHot(c);
        var h1 = forwardpassLayer(x, lstm.L1, S1, lstm.num_cells);
        var h2 = forwardpassLayer(h1, lstm.L2, S2, lstm.num_cells);
        var h3 = forwardpassLayer(h2, lstm.L3, S3, lstm.num_cells);
        var y = lstm.Wyh.dot(h3).plus(lstm.by);
        var p = y.mulEach(t).map(Math.exp);
        return fromDist(p.mulEach(1 / p.getSum()));
    }

    function forwardpassLayer(x, L, S, nc) {

        var Sa = S.toArray()[0];

        // Calc all gates, input values
        //z = W*[x;h] + b;
        var z = L.W.dot(new Matrix(x.trans().toArray()[0].concat(Sa.slice(5 * nc, 6 * nc + 1))).trans()).plus(L.b);
        var za = z.toArray();

        // a = tanh(z(1:num_cells));
        var a = new Matrix(za.slice(0 * nc, nc)).map(Math.tanh);

        // i = lsig(z(1 + num_cells : 2 * num_cells));
        var i = new Matrix(za.slice(1 * nc, 2 * nc)).sigmoid();

        // f = lsig(z(1 + 2 * num_cells : 3 * num_cells));
        var f = new Matrix(za.slice(2 * nc, 3 * nc)).sigmoid();

        // o = lsig(z(1 + 3 * num_cells : 4 * num_cells));
        var o = new Matrix(za.slice(3 * nc, 4 * nc)).sigmoid();

        // % Calc cell state
        // % note - cell state is linear (not tanh'd)
        // c = a .* i + cp .* f;
        var cp = new Matrix(Sa.slice(4 * nc, 5 * nc)).trans();
        var c = a.mul(i).plus(cp.mul(f));

        // h = tanh(c) .* o;
        var h = c.map(Math.tanh).mul(o);

        Sa = a.trans().toArray()[0]
            .concat(i.trans().toArray()[0])
            .concat(f.trans().toArray()[0])
            .concat(o.trans().toArray()[0])
            .concat(c.trans().toArray()[0])
            .concat(h.trans().toArray()[0]);

        var S_ = new Matrix(Sa);
        S.data = S_.data;
        return h;
    }
}

console.log(sample(140, 1.5));