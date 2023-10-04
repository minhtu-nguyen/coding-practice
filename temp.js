const division = (numberator, denominator, callback) => {
    if (denominator === 0) {
        callback(new Error("Divide by zero error!"));
    } else {
        callback(null, numberator / denominator);
    }
};

division(5, 1, (err, result) => {
    if (err) {
        return console.log(err.message);
    }
    console.log(`Result: ${result}`);
})