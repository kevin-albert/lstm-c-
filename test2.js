var lstm;
if (module && module.exports) lstm = module.exports;
else {
    window.lstm = {};
    lstm = window.lstm;
}

(function(o) {
    o.foo = 'bar';
})(lstm);
