#include "leetcode.h"
/*
给定一个字符串，逐个翻转字符串中的每个单词。

示例：

输入: ["t","h","e"," ","s","k","y"," ","i","s"," ","b","l","u","e"]
输出: ["b","l","u","e"," ","i","s"," ","s","k","y"," ","t","h","e"]
注意：

单词的定义是不包含空格的一系列字符
输入字符串中不会包含前置或尾随的空格
单词与单词之间永远是以单个空格隔开的
进阶：使用 O(1) 额外空间复杂度的原地解法。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/reverse-words-in-a-string-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

void reverse(char *s, int len)
{
    char *t = s + len - 1;
    char tmp;
    while (s < t) {
        tmp = *s;
        *s = *t;
        *t = tmp;

        s++;
        t--;
    }
}

static char *p1, *p2;

void reverseWords(char* s, int sSize){
    if (s == NULL || sSize <= 1)
        return ;

    p1 = s;
    p2 = s - 1;

    for (int i = 0; i < sSize; i++) {
        if (s[i] != ' ')
            continue;

        /* s[i] == ' ' */
        p1 = p2 + 1;
        p2 = &s[i];

        reverse(p1, (p2 - p1));
    }

    /* p1 to end */
    p1 = p2 + 1;
	/* I firstly implemented this and found we need to reverse all once more...*/
    reverse(p1, sSize - (p1 - s));
	reverse(s, sSize);
}

int main(void)
{
	char str[16] = "the sky is bule";
	reverseWords(str, 15);
	printf("%s\n", str);
	return 0;
}