#include "device_control.hpp"
#include <stdio.h>
#include <stdlib.h>

static int save_log(char *filename, char *log);

static int save_log(char *filename, char *log)
{
    FILE *fp;
    fp = fopen(filename, "w");
    if (fp == NULL){
        printf("file open error\n");
        exit(1);
    }
    fprintf(fp, "%s", log);
    fclose(fp);

    return 0;
}

int device_control_main(void)
{
    char filename[256];
    int ii;
    printf("Hello, next world\n");
    for(ii=0; ii<5; ii++){
        sprintf(filename, "file_%d.txt", ii);
        // printf("%s\n", filename);
        save_log(filename, filename);
    }
    
    return 0;
}
