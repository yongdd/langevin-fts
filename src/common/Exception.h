#ifndef FTS_Exception_H_
#define FTS_Exception_H_

#include <string>
#include <stdexcept>

class Exception : public std::runtime_error {
public:
    Exception(const std::string &arg, const char *file, int line, const char *func):
        std::runtime_error("File: \"" + std::string(file) + "\", line: " + std::to_string(line) + ", function <" + std::string(func) + ">\n\t" + arg){};
    Exception(const std::string &arg, const char *file, const char *func):
        std::runtime_error("File: \"" + std::string(file) + "\", function <" + std::string(func) + ">\n\t" + arg){};
    ~Exception() throw() {};
};
#define throw_with_line_number(arg) throw Exception(arg, __FILE__, __LINE__, __func__);
#define throw_without_line_number(arg) throw Exception(arg, __FILE__, __func__);

#endif