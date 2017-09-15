from blessings import Terminal
import progressbar
import sys


class TermLogger(object):
    def __init__(self, n_epochs, train_size, test_size):
        self.n_epochs = n_epochs
        self.train_size = train_size
        self.test_size = test_size
        self.t = Terminal()
        s = 10
        e = 1   # epoch bar position
        tr = 3  # train bar position
        ts = 6  # test bar position
        h = self.t.height

        for i in range(10):
            print('')
        self.epoch_bar = progressbar.ProgressBar(max_value=n_epochs, fd=Writer(self.t, (0, h-s+e)))

        self.train_writer = Writer(self.t, (0, h-s+tr))
        self.train_bar_writer = Writer(self.t, (0, h-s+tr+1))

        self.test_writer = Writer(self.t, (0, h-s+ts))
        self.test_bar_writer = Writer(self.t, (0, h-s+ts+1))

        self.reset_train_bar()
        self.reset_test_bar()

    def reset_train_bar(self):
        self.train_bar = progressbar.ProgressBar(max_value=self.train_size, fd=self.train_bar_writer)

    def reset_test_bar(self):
        self.test_bar = progressbar.ProgressBar(max_value=self.test_size, fd=self.test_bar_writer)


class Writer(object):
    """Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """

    def __init__(self, t, location):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.location = location
        self.t = t

    def write(self, string):
        with self.t.location(*self.location):
            sys.stdout.write("\033[K")
            print(string)

    def flush(self):
        return