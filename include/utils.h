#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>

namespace coalsack
{
    [[maybe_unused]] static void write_string(std::ostream &ss, std::string value)
    {
        uint32_t length = (uint32_t)value.size();
        ss.write((const char *)&length, sizeof(length));
        ss.write((const char *)&value[0], length);
    }

    [[maybe_unused]] static void write_int8(std::ostream &ss, int8_t value)
    {
        ss.write((const char *)&value, sizeof(value));
    }

    [[maybe_unused]] static void write_uint8(std::ostream &ss, uint8_t value)
    {
        ss.write((const char *)&value, sizeof(value));
    }

    [[maybe_unused]] static void write_int16(std::ostream &ss, int16_t value)
    {
        ss.write((const char *)&value, sizeof(value));
    }

    [[maybe_unused]] static void write_uint16(std::ostream &ss, uint16_t value)
    {
        ss.write((const char *)&value, sizeof(value));
    }

    [[maybe_unused]] static void write_int32(std::ostream &ss, int32_t value)
    {
        ss.write((const char *)&value, sizeof(value));
    }

    [[maybe_unused]] static void write_uint32(std::ostream &ss, uint32_t value)
    {
        ss.write((const char *)&value, sizeof(value));
    }

    [[maybe_unused]] static void write_int64(std::ostream &ss, int64_t value)
    {
        ss.write((const char *)&value, sizeof(value));
    }

    [[maybe_unused]] static void write_uint64(std::ostream &ss, uint64_t value)
    {
        ss.write((const char *)&value, sizeof(value));
    }

    [[maybe_unused]] static void write_single(std::ostream &ss, float value)
    {
        ss.write((const char *)&value, sizeof(value));
    }

    [[maybe_unused]] static void write_double(std::ostream &ss, double value)
    {
        ss.write((const char *)&value, sizeof(value));
    }

    [[maybe_unused]] static std::string read_string(std::istream &ss)
    {
        uint32_t length = 0;
        ss.read((char *)&length, sizeof(uint32_t));

        std::vector<char> data(length);
        ss.read((char *)&data[0], length);

        return std::string(data.begin(), data.end());
    }

    [[maybe_unused]] static uint16_t read_uint16(std::istream &ss)
    {
        uint16_t value;
        ss.read((char *)&value, sizeof(value));
        return value;
    }

    [[maybe_unused]] static uint32_t read_uint32(std::istream &ss)
    {
        uint32_t value;
        ss.read((char *)&value, sizeof(value));
        return value;
    }

    [[maybe_unused]] static int32_t read_int32(std::istream &ss)
    {
        int32_t value;
        ss.read((char *)&value, sizeof(value));
        return value;
    }

    [[maybe_unused]] static int64_t read_int64(std::istream &ss)
    {
        int64_t value;
        ss.read((char *)&value, sizeof(value));
        return value;
    }

    [[maybe_unused]] static float read_single(std::istream &ss)
    {
        float value;
        ss.read((char *)&value, sizeof(value));
        return value;
    }

    [[maybe_unused]] static double read_double(std::istream &ss)
    {
        double value;
        ss.read((char *)&value, sizeof(value));
        return value;
    }
}
