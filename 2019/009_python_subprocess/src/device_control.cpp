#include "device_control.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// define
#define COMMUNICATE_BUF_MAX_LEN 256
#define EOF_STRINGS "EOFEOF"
#define STR_COMP_SAME_VALUE 0

static int save_log(char *filename, char *log);

static char *get_char_from_stdin(char *s, int n);

static char *get_char_from_stdin(char *s, int n)
{
   // get all strings.
   if(fgets(s, n, stdin) == NULL) return NULL;
   // get linefeed code
   char *ln = strchr(s, '\n');
   // replace linefeed code to tarmination character.
   if(ln){
       *ln = '\0';
   }
   else{
       printf("invalid stdin strings...\n");
       printf("linefeed code was not found.\n");
       exit(1);
   }
   return s;
}

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
    char buf[COMMUNICATE_BUF_MAX_LEN];
    int counter = 0;
    // int ii;
    printf("Hello, next world\n");
    while(1){
        // buffer check
        if(get_char_from_stdin(buf, COMMUNICATE_BUF_MAX_LEN) == NULL){ continue; }
        if(strcmp(buf, EOF_STRINGS) == STR_COMP_SAME_VALUE){ break; }

        // exec commands
        sprintf(filename, "file_%d.txt", counter);
        save_log(filename, buf);
        counter += 1;
    }

    printf("exit while loop\n");
    
    return 0;
}
