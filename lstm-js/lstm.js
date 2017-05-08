var S1 = Vector.zero(num_cells * 6);
var S2 = Vector.zero(num_cells * 6);
var S3 = Vector.zero(num_cells * 6);

var c = '^';

function sample(t) {

    var limit = 280;
    var output = '';
    while (1) {
        c = forwardpass(c, t);
        if (c == '^' || --limit == 0) break;
        output += c;
    }

    return output;

    function toOneHot(c) {
        var v = Vector.zero(num_input);
        v.data[0][encode(c)] = 1;
        return v.trans();
    }

    function fromDist(p) {
        var rand = Math.random();
        var a = p.trans().toArray()[0];
        for (var i = 0; i < a.length; ++i) {
            rand -= a[i];
            if (rand <= 0) {
                return decode(i);
            }
        }
        return decode(a[a.length]);
    }

    function forwardpass(c, t) {
        var x = toOneHot(c);
        var h1 = forwardpassLayer(x, L1, S1, num_cells);
        var h2 = forwardpassLayer(h1, L2, S2, num_cells);
        var h3 = forwardpassLayer(h2, L3, S3, num_cells);
        var y = Wyh.dot(h3).plus(by);
        var p = y.mulEach(1 / t).map(Math.exp);
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

function reset() {
    S1 = Vector.zero(num_cells * 6);
    S2 = Vector.zero(num_cells * 6);
    S3 = Vector.zero(num_cells * 6);
}

self.addEventListener('message', function (e) {
    var data = e.data;
    if (typeof data == 'string')
        data = JSON.parse(data);

    switch (data.cmd) {
        case 'sample':
            console.log('sample ' + data.t);
            var t = typeof data.t == 'number' ? data.t : 1.5;
            let tweet = '';
            while (tweet.trim().length < 3) {
                tweet = sample(t);
            }
            self.postMessage({
                event: 'tweet',
                data: tweet
            });
            break;
        case 'reset':
            console.log('reset');
            reset();
            break;
        default:
            console.error('wat', data);
            break;
    }
    var s = sample(e.t);
}, false);

self.postMessage({
    event: 'ready'
});