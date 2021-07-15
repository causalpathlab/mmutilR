#include "std_util.hh"

char *
str2char(const std::string &s)
{
    char *ret = new char[s.size() + 1];
    std::strcpy(ret, s.c_str());
    return ret;
}

std::vector<std::string>
split(const std::string &s, char delim)
{
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        elems.push_back(std::move(item));
    }
    return elems;
}
