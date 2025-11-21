#include "leetcode.h"

/* 在由 1 x 1 方格组成的 N x N 网格 grid 中，每个 1 x 1 方块由 /、\ 或空格构成。这些字符会将方块划分为一些共边的区域。
（请注意，反斜杠字符是转义的，因此 \ 用 "\\" 表示。）。
返回区域的数目。

示例 1：
输入：
[
  " /",
  "/ "
]
输出：2
解释：2x2 网格如下：
| /|
|/ |

示例 2：
输入：
[
  " /",
  "  "
]
输出：1
解释：2x2 网格如下：
| /|
|  |

示例 3：
输入：
[
  "\\/",
  "/\\"
]
输出：4
解释：（回想一下，因为 \ 字符是转义的，所以 "\\/" 表示 \/，而 "/\\" 表示 /\。）
2x2 网格如下：
|\/|
|/\|

示例 4：
输入：
[
  "/\\",
  "\\/"
]
输出：5
解释：（回想一下，因为 \ 字符是转义的，所以 "/\\" 表示 /\，而 "\\/" 表示 \/。）
2x2 网格如下：
|/\|
|\/|

示例 5：
输入：
[
  "//",
  "/ "
]
输出：3
解释：2x2 网格如下：
|//|
|/ |

 
提示：
1 <= grid.length == grid[0].length <= 30
grid[i][j] 是 '/'、'\'、或 ' '。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/regions-cut-by-slashes
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

static int *fa;
static int faSize;
static int gSize;
int find(int u)
{
    return (fa[u] == u) ? u : (fa[u] = find(fa[u]));
}

void unin(int u, int v)
{
    fa[find(u)] = find(v);
}

int top(int i, int j)
{
    return i * gSize * 4 + j * 4;
}

int rig(int i, int j)
{
    return i * gSize * 4 + j * 4 + 1;
}

int bot(int i, int j)
{
    return i * gSize * 4 + j * 4 + 2;
}

int lef(int i, int j)
{
    return i * gSize * 4 + j * 4 + 3;
}

int compare(const void *a, const void *b)
{
    return *(int *)a - *(int *)b;
}

int regionsBySlashes(char ** grid, int gridSize){
    /* init fa */
    int count = 1;

    if (grid == NULL || gridSize <= 0)
        return 0;

    /* each region is divided into 4 parts */
    faSize = gridSize * gridSize * 4;
    fa = malloc(sizeof(int) * faSize);
    gSize = gridSize;

    for (int i = 0; i < faSize; i++) {
        fa[i] = i;
    }

    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            switch (grid[i][j]) {
                int curIdx = i * gridSize * 4 + j * 4;
                case '/': 
                    unin(top(i, j), lef(i, j));
                    unin(rig(i, j), bot(i, j));
                    break;
                case '\\':
                    unin(top(i, j), rig(i, j));
                    unin(lef(i, j), bot(i, j));
                    break;
                case ' ':
                    unin(top(i, j), rig(i, j));
                    unin(top(i, j), lef(i, j));
                    unin(lef(i, j), bot(i, j));
                    break;
                default:
                    printf("error\n");
                    return -1;
            }

            if (i > 0) {
                unin(top(i, j), bot(i - 1, j));
            }

            if (j > 0) {
                unin(lef(i, j), rig(i, j - 1));
            }
        }
    }

    for (int i = 0; i < faSize; i++) {
        /* zip all */
        (void)find(i);
    }

    qsort(fa, faSize, sizeof(int), compare);
    for (int i = 1; i < faSize; i++) {
        if (fa[i - 1] != fa[i]) {
            count++;
        }
    }

    free(fa);

    return count;
}

int main(void)
{
#define CASE_NR 2
    char **grid = malloc(sizeof(char *) * CASE_NR);
    int ret;
    for (int i = 0; i < CASE_NR; i++) {
        grid[i] = malloc(CASE_NR);
    }

    grid[0][0] = '\\';
    grid[0][1] = '/';
    grid[1][0] = '\\';
    grid[1][1] = '/';
    /* result is 4 */

    ret = regionsBySlashes(grid, CASE_NR);
    printf("%d\n", ret);

    for (int i = 0; i < CASE_NR; i++) {
        free(grid[i]);
    }
    free(grid);
    return 0;
}