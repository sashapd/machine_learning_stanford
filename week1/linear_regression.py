class LinearRegression:
    @staticmethod
    def hypothesis(theta0, theta1, x):
        return theta0 + theta1 * x

    def cost_function_derived(self, theta0, theta1, power):
        sum = 0
        m = len(self.X)
        for x, y in zip(self.X, self.Y):
            sum += (self.hypothesis(theta0, theta1, x) - y) * (x ** power)
        return sum / m

    def fit(self, X, Y, learning_rate):
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate

        self.theta0 = 0
        self.theta1 = 0

        updatedTheta0 = self.theta0 - self.learning_rate * self.cost_function_derived(self.theta0, self.theta1, 0)
        updatedTheta1 = self.theta1 - self.learning_rate * self.cost_function_derived(self.theta0, self.theta1, 1)

        epsilon = 0.000000000001

        while abs(self.theta0 - updatedTheta0) > epsilon or abs(self.theta1 - updatedTheta1) > epsilon:
            self.theta0 = updatedTheta0
            self.theta1 = updatedTheta1
            updatedTheta0 = self.theta0 - self.learning_rate * self.cost_function_derived(self.theta0, self.theta1, 0)
            updatedTheta1 = self.theta1 - self.learning_rate * self.cost_function_derived(self.theta0, self.theta1, 1)

    def predict(self, input):
        result = []
        for x in input:
            prediction = self.hypothesis(self.theta0, self.theta1, x)
            result.append(prediction)
        return result

#Example: staight line
#y = 2 + 3x
x = [0, 3, 4]
y = [2, 11, 14]
learning_rate = 0.1

regression = LinearRegression()
regression.fit(x, y, learning_rate)

print("prediction", regression.predict([1, 2, 5, 42, 100])) #[4.999999999978103, 7.999999999987098, 17.000000000014087, 128.00000000034694, 302.00000000086874]
print("slope: ", regression.theta1) #slope:  3.000000000008996
print("intercept: ", regression.theta0) #intercept:  1.9999999999691067
