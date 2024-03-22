## 使用sanitizer检查内存泄漏

CMakeLists.txt编写如下:

```bash
cmake_minimum_required(VERSION 3.0)
project(MyProject)
set(CMAKE_CXX_STANDARD 11)
# 需要加上-g -fsanitize=leak编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=leak")
add_executable(MyExecutable main.cpp)
```

示例代码:

```C++
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace std;
 
void test(){
       printf("test.\n");
       int *ptr = new int[5];
       ptr[10] = 7;                              // 越界访问
       delete [] ptr;
       char *ps;
       ps = new char[10];                        // 指针未释放
       strcpy(ps, "ABC");
}
 
int main(){
       test();
       return 0;
}
```

编译执行后结果如下:

![](assets/asan.jpg)