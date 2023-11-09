#include "io.hh"

int
read_line_file(const std::string filename,
               std::vector<std::string> &in,
               const std::size_t max_word = 1,
               const char sep = '_')
{
    int ret = EXIT_SUCCESS;

    if (is_file_gz(filename)) {
        igzstream ifs(filename.c_str(), std::ios::in);
        ret = read_line_stream(ifs, in, max_word, sep);
        ifs.close();
    } else {
        std::ifstream ifs(filename.c_str(), std::ios::in);
        ret = read_line_stream(ifs, in, max_word, sep);
        ifs.close();
    }

    return ret;
}

bool
file_exists(std::string filename)
{
    if (filename.size() < 1)
        return false;
    std::ifstream f(filename.c_str());
    return f.good();
}

bool
all_files_exist(std::vector<std::string> filenames)
{
    bool ret = true;
    for (auto f : filenames) {
        if (!file_exists(f)) {
            TLOG(std::left << std::setw(10) << "Missing: " << std::setw(30)
                           << f);
            ret = false;
        }
        TLOG(std::left << std::setw(10) << "Found: " << std::setw(30) << f);
    }
    return ret;
}

void
copy_file(const std::string _src, const std::string _dst)
{
    std::ifstream src(_src.c_str(), std::ios::binary);
    std::ofstream dst(_dst.c_str(), std::ios::binary);
    dst << src.rdbuf();
}

void
remove_file(const std::string _file)
{
    if (file_exists(_file)) {
        std::remove(_file.c_str());
    }
}

void
rename_file(const std::string _src, const std::string _dst)
{
    std::rename(_src.c_str(), _dst.c_str());
}

/////////////////////////////////
// common utility for data I/O //
/////////////////////////////////

bool
is_file_gz(const std::string filename)
{
    if (filename.size() < 3)
        return false;
    return filename.substr(filename.size() - 3) == ".gz";
}

bool
is_file_bgz(const std::string filename)
{
    if (bgzf_is_bgzf(filename.c_str()) < 1)
        return false;
    return true;
}

std::shared_ptr<std::ifstream>
open_ifstream(const std::string filename)
{
    std::shared_ptr<std::ifstream> ret(
        new std::ifstream(filename.c_str(), std::ios::in));
    return ret;
}

std::shared_ptr<igzstream>
open_igzstream(const std::string filename)
{
    std::shared_ptr<igzstream> ret(
        new igzstream(filename.c_str(), std::ios::in));
    return ret;
}
