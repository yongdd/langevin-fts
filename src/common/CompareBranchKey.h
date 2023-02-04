/*----------------------------------------------------------
* This file defines comparision function for branched key
*-----------------------------------------------------------*/

#ifndef COMPARE_BRANCH_KEY_H_
#define COMPARE_BRANCH_KEY_H_

#include <string>
#include <vector>
#include <algorithm>

struct CompareBranchKey
{
    bool operator()(const std::string& str1, const std::string& str2)
    {
        int mix_length = std::min(str1.length(), str2.length());
        for(int i=0; i<mix_length; i++)
        {
            if (str1[i] == str2[i])
                continue;
            else if (str2[i] == '[')
                return true;
            else if (str1[i] == '[')
                return false;
            else if (str2[i] == ']')
                return true;
            else if (str1[i] == ']')
                return false;
            else if (str2[i] == '(')
                return true;
            else if (str1[i] == '(')
                return false;
            else if (str2[i] == ')')
                return true;
            else if (str1[i] == ')')
                return false;
            else
                return str1[i] < str2[i];
        }
        return str1.length() < str2.length();
    }
};
#endif
