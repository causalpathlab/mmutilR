#include "util.hh"

std::string
curr_time()
{
    time_t rawtime;
    time(&rawtime);
    struct tm *timeinfo = localtime(&rawtime);
    char buff[80];
    strftime(buff, 80, "%c", timeinfo);
    return std::string(buff);
}

std::string
zeropad(const int t, const int tmax)
{
    std::string tt = std::to_string(t);
    std::string ttmax = std::to_string(tmax);
    const int ndigit = ttmax.size();

    std::ostringstream ss;
    ss << std::setw(ndigit) << std::setfill('0') << tt;
    return std::string(ss.str());
}
