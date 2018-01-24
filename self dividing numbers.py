class Solution(object):
    def selfDividingNumbers(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """
        def selfdividing(x):
            x_str = str(x)
            for each in x_str:
                if (each == '0' or x % int(each) !=0 ):
                    return False
            return True
                
        result = []
        for i in range(left,right+1):
            if (selfdividing(i)):
                result.append(i)
        return result

#another excellent solution
class Solution(object):
def selfDividingNumbers(self, left, right):
    is_self_dividing = lambda num: '0' not in str(num) and all([num % int(digit) == 0 for digit in str(num)])
    return filter(is_self_dividing, range(left, right + 1))
