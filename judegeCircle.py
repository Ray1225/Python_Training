class Solution(object):
    def judgeCircle(self, moves):
        """
        :type moves: str
        :rtype: bool
        """
        self.x = 0
        self.y = 0
        for each in moves:
            if each  == 'L':
                self.x -= 1
            elif each == 'R':
                self.x += 1
            elif each == 'U':
                self.y += 1
            elif each == 'D':
                self.y -= 1
        if (self.x == 0 and self.y == 0):
            return True
        else:
            return False
