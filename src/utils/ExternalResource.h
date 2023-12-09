#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

template <typename T>
concept readwrite_defined_c = requires(T t) {
	{ std::cout << t };
	{ std::cin >> t };
};
template <typename>
constexpr bool readwrite_defined_v = false;
template <readwrite_defined_c T>
constexpr bool readwrite_defined_v<T> = true;

template <unsigned N>
struct string_literal {
	constexpr string_literal(const char(&str)[N]) {
		std::copy_n(str, N, value);
	}
	constexpr string_literal() { };
	char value[N]{};
};

template <unsigned N1, unsigned N2>
struct concat_strings : public string_literal<N1 + N2 - 1> {
	inline static constexpr unsigned N = N1 + N2 - 1;
	constexpr concat_strings(const char(&str1)[N1], const char(&str2)[N2]) {
		std::copy_n(str1, sizeof(str1), this->value);
		std::copy_n(str2, sizeof(str2), this->value + sizeof(str1) - 1);
	}
	constexpr operator string_literal<N>() const {
		string_literal<N> res;
		std::copy_n(this->value, N, res.value);
		return res;
	}
};

template <string_literal Path, bool IsBinary = false>
struct external_resource_file : std::fstream {
	external_resource_file();

	template <typename T>
	bool readObject(T* object);

	template <typename T>
	void writeObject(T* value);

	~external_resource_file() { close(); }
};

template <string_literal Path, typename T, auto...>
struct external_resource_object {
	T* value = new T();
	external_resource_file<Path, not readwrite_defined_v<T>> file;
	external_resource_object() { file.readObject(value); }
	~external_resource_object() { file.writeObject(value); }
};

template <string_literal Path, typename T, auto...>
struct external_resource {
	inline static external_resource_object<Path, T> object;
	inline static T& value = *object.value;
};
template <string_literal Path, typename T, auto...>
T& external_resource_v = external_resource<Path, T>::value;

template<string_literal Path, bool IsBinary>
template<typename T>
inline bool external_resource_file<Path, IsBinary>::readObject(T* object) {
	if (!is_open()) return false;
	if constexpr (IsBinary)
		read((char*)object, sizeof(T));
	else
		*this >> *object;
	return true;
}

template<string_literal Path, bool IsBinary>
template<typename T>
inline void external_resource_file<Path, IsBinary>::writeObject(T* value) {
	std::ofstream ofs(Path.value, (IsBinary) ? std::ios::binary : std::ios::out);

	if (!ofs.is_open()) {
		std::string fullPath(Path.value);
		size_t fileStartIndex = fullPath.find_last_of("/\\");
		if (fileStartIndex == std::string::npos)
			fileStartIndex = 0;

		if (fileStartIndex != 0)
			std::filesystem::create_directories(fullPath.substr(0, fileStartIndex));

		ofs.open(Path.value, (IsBinary) ? std::ios::binary : std::ios::out);
	}

	if constexpr (IsBinary)
		ofs.write((char*)value, sizeof(T));
	else
		ofs << *value;
	ofs.close();
}

template<string_literal Path, bool IsBinary>
inline external_resource_file<Path, IsBinary>::external_resource_file() : std::fstream(Path.value, ios_base::in | ios_base::out | ((IsBinary) ? std::ios::binary : 0)) {
	std::cout << (is_open() ? "Opened" : "Coundn't open") << (IsBinary ? " binary " : " ") << "file " << Path.value << std::endl;
}
