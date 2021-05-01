#include "Settings.h"
#include <fstream>
#include <algorithm>
#include <regex>
#include <string>

Settings::Settings(const string &path) {
    fstream inStream(path);
    std::string str((std::istreambuf_iterator<char>(inStream)),std::istreambuf_iterator<char>());
    str.erase(remove_if(str.begin(), str.end(), ::isspace), str.end());

    this->leaders =         std::stoi(getAtribVal(str, "leaders"));
    this->followers =       std::stoi(getAtribVal(str, "followers"));
    this->heightmapGridX =  std::stoi(getAtribVal(str, "heightmapGridX"));
    this->heightmapGridY =  std::stoi(getAtribVal(str, "heightmapGridY"));
    this->viewportWidth =   std::stoi(getAtribVal(str, "viewportWidth"));
    this->viewportHeight =  std::stoi(getAtribVal(str, "viewportHeight"));
    this->leaderRadius =    std::stof(getAtribVal(str, "leaderRadius"));
    this->speedFactor =     std::stof(getAtribVal(str, "speedFactor"));
    this->heightMap =       getAtribVal(str, "heightmap");
    this->outputFile =      getAtribVal(str, "outputFile");
}

string Settings::getAtribVal(string &str, const string &atribName) {
    smatch m;
    regex rt(atribName + "\":([\\S\\s]+?(?=,|}))");
    regex_search(str, m, rt);
    string val = m.str().substr(atribName.size()+2);
    if (val.at(0) == '\"') {
        val = val.substr(1, val.size()-2);
    }
    return val;
}
