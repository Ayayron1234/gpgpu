#pragma once
#include <iostream>
#include <map>
#include <tuple>
#include <type_traits>
#include <vector>
#include <string>
#include <format>
#include <format>

#define ASSERT(_e)\
	if (!(_e)) {\
		std::cout << "Assertion faled at " << __FILE__ << "(" << __LINE__ << "):\n   ASSERT(" << #_e << ")\n" << std::endl;\
		__debugbreak();\
	}

#define ASSERT_M(_e, m)																													\
	if (!(_e)) {																														\
		std::cout << "Assertion faled at " << __FILE__ << "(" << __LINE__ << "):\n   ASSERT(" << #_e << ")\n   " << m << std::endl;		\
		__debugbreak();																													\
	}

#define JSON_C(ClassName, ...)\
template <> \
struct Json::ReflectionData<ClassName> {\
	using Type = ClassName;\
	constexpr static bool defined = true;\
	constexpr static auto properties = std::make_tuple(__VA_ARGS__);\
};

#define JSON_C_TEMPLATE(ClassName, ...)\
template <typename T>\
struct Json::ReflectionData<ClassName<T>> {\
	using Type = ClassName<T>;\
	constexpr static bool defined = true;\
	constexpr static auto properties = std::make_tuple(__VA_ARGS__);\
};

#define JSON_M(MemberName) Json::property(&Type:: ## MemberName ## , #MemberName)

class Json;

class Json {
public:
	//template <typename T, unsigned N>
	//using get_member_t = std::remove_reference_t<decltype(std::get<N>(Json::ReflectionData<T>::properties))>::Type;

	template <typename>
	inline static constexpr size_t array_size = 0;

	template <typename T, size_t N>
	inline static constexpr size_t array_size<T[N]> = N;

	template <typename T, T... S, typename F>
	static constexpr void for_sequence(std::integer_sequence<T, S...>, F&& f) {
		(static_cast<void>(f(std::integral_constant<T, S>{})), ...);
	}

	enum class DataType { NONE = 0x00, OBJECT, ARRAY, NUMBER, STRING };

	DataType					dataType = DataType::NONE;
	std::map<std::string, Json> object;
	std::vector<Json>			array;
	long double					number = 0;
	std::string					string;

	// TODO: store a pointer to the typeid(T).name() of the type the json was converted from
	const char* typeNamePtr = nullptr;

	Json() { }
	template <typename T>
		requires(not std::is_same_v<T, Json>)
	explicit Json(const T& v) { Json json; json = v; *this = json; }

	static Json CreateEmptyObject();

	template <typename T>
	static Json CreateEmptyObject();

	static Json CreateEmptyArray();

	/* ------------------------ *
	*		MEMBER ACCESS		*
	* ------------------------- */

	Json& operator[](std::string name);

	Json& operator[](const char* name);

	const Json& operator[](std::string name) const;

	const Json& operator[](const char* name) const;

	template <typename T>
		requires(std::is_arithmetic_v<T> or std::is_enum_v<T>)
	Json& operator[](T index);

	template <typename T>
		requires(std::is_arithmetic_v<T> or std::is_enum_v<T>)
	const Json& operator[](T index) const;

	/* ---------------- *
	*		TO JSON		*
	* ----------------- */

	template <typename T>
	Json& operator=(T);

	template <typename T>
		requires requires(T t, Json json) { { t.operator Json() }; { t = json }; }
	Json& operator=(T);

	template <typename T>
		requires(std::is_array_v<T>)
	Json& operator=(T);

	template <typename T>
		requires(std::_Is_specialization_v<T, std::vector>)
	Json& operator=(T);

	template <typename T>
		requires(std::is_arithmetic_v<T> or std::is_enum_v<T>)
	Json& operator=(T);

	template <>
	Json& operator=(std::string);

	template <>
	Json& operator=(std::map<std::string, Json>);

	template <>
	Json& operator=(const char*);

	/* -------------------- *
	*		FROM JSON		*
	* --------------------- */

	template <typename T>
	operator T() const;

	template <typename T>
		requires(std::is_array_v<T>)
	operator T() const;

	template <typename T>
		requires(std::_Is_specialization_v<T, std::vector>)
	operator T() const;

	template <typename T>
		requires(std::is_arithmetic_v<T> or std::is_enum_v<T>)
	operator T() const;

	operator std::string() const;

	operator std::map<std::string, Json>() const;

	/* ---------------------------- *
	*		TYPELESS CONVERSION		*
	* ----------------------------- */

	using converter_function_t = Json(const unsigned char*);
	using parser_function_t = void(const Json& object, void* dst);

	template <typename T>
	static converter_function_t* GetConverter();

	template <typename T>
	static parser_function_t* GetParser();

	/* ---------------- *
	*		WRAPPER		*
	* ----------------- */

	template <typename T>
	struct wrap : public T {
		wrap() { }
		wrap(const T& value) : T(value) {}
	};

private:

	/* -------------------- *
	*		REFLECTION		*
	* --------------------- */

	template<typename Class, typename T>
	struct PropertyImpl {
		using Type = T;

		constexpr PropertyImpl(T Class::* aMember, const char* aName)
			: member{ aMember }
			, name{ aName }
		{ }

		T Class::* member;
		const char* name;
	};

	template<typename Class, typename T>
	static constexpr auto property(T Class::* member, const char* name) {
		return PropertyImpl<Class, T>{member, name};
	}

	template <typename T>
	class ReflectionData {
	public:
		using Type = T;
		constexpr static bool defined = false;
	};

	/* ---------------- *
	*		UTILS		*
	* ----------------- */

	template <typename T>
	static void setArray(const Json& value, T* arrayPtr, size_t elementSize, size_t elementCount) {
		for (size_t i = 0; i < elementCount; ++i)
			arrayPtr[i] = value.array.at(i);
	}

	template <typename T>
	static Json anyToJson(const unsigned char* value);

	template <typename T>
	static void jsonToAny(const Json& value, void* dst);

	inline static bool s_indentOutput = true;

	friend std::ostream& operator<<(std::ostream&, const Json&);
};

inline Json Json::CreateEmptyObject() {
	Json json;
	json.dataType = DataType::OBJECT;
	return json;
}

template<typename T>
inline Json Json::CreateEmptyObject() {
	Json json;
	json.dataType = DataType::OBJECT;
	json.typeNamePtr = typeid(T).name();
	return json;
}

inline Json Json::CreateEmptyArray() {
	Json json;
	json.dataType = DataType::ARRAY;
	return json;
}

/* ------------------------ *
*		MEMBER ACCESS		*
* ------------------------- */

inline Json& Json::operator[](std::string name) {
	ASSERT(dataType == DataType::OBJECT);
	return object[std::move(name)];
}

inline Json& Json::operator[](const char* name) {
	ASSERT(dataType == DataType::OBJECT);
	return object[name];
}

inline const Json& Json::operator[](std::string name) const {
	ASSERT(dataType == DataType::OBJECT);

	auto it = object.find(std::move(name));
	if (it != object.end())
		return it->second;
	throw;
}

inline const Json& Json::operator[](const char* name) const {
	ASSERT(dataType == DataType::OBJECT);

	auto it = object.find(name);
	if (it != object.end())
		return it->second;
	throw;
}

template <typename T>
	requires(std::is_arithmetic_v<T> or std::is_enum_v<T>)
inline Json& Json::operator[](T index) {
	ASSERT(dataType == DataType::ARRAY);
	return array.at(index);
}

template <typename T>
	requires(std::is_arithmetic_v<T> or std::is_enum_v<T>)
inline const Json& Json::operator[](T index) const {
	ASSERT(dataType == DataType::ARRAY);
	return array.at(index);
}

/* ---------------- *
*		TO JSON		*
* ----------------- */

template <typename T>
inline Json& Json::operator=(T value) {
	static_assert(Json::ReflectionData<T>::defined, "Use the JSON_C and JSON_M macros to define the members for type T.");

	dataType = DataType::OBJECT;
	typeNamePtr = typeid(T).name();

	constexpr size_t memberCount = std::tuple_size<decltype(Json::ReflectionData<T>::properties)>::value;
	for_sequence(std::make_index_sequence<memberCount>{}, [&](auto i) {
		constexpr auto property = std::get<i>(Json::ReflectionData<T>::properties);
		using Type = typename decltype(property)::Type;

		object[property.name].operator=<Type>(value.*(property.member));
		});

	return *this;
}

template <typename T>
	requires requires(T t, Json json) { { t.operator Json() }; { t = json }; }
Json& Json::operator=(T value) {
	*this = value.operator Json();
	return *this;
}

template <typename T>
	requires(std::is_array_v<T>)
inline Json& Json::operator=(T value) {
	dataType = DataType::ARRAY;
	for (size_t i = 0; i < array_size<T>; i++)
		array.push_back((Json)value[i]);
	return *this;
}

template <typename T>
	requires(std::_Is_specialization_v<T, std::vector>)
inline Json& Json::operator=(T value) {
	dataType = DataType::ARRAY;
	for (size_t i = 0; i < value.size(); i++)
		array.push_back(value.at(i));
	return *this;
}

template <typename T>
	requires(std::is_arithmetic_v<T> or std::is_enum_v<T>)
inline Json& Json::operator=(T value) {
	dataType = DataType::NUMBER;
	typeNamePtr = typeid(T).name();
	if constexpr (std::is_enum_v<T>)
		number = static_cast<int>(value);
	else
		number = value;
	return *this;
}

template <>
inline Json& Json::operator=(std::string value) {
	dataType = DataType::STRING;
	string = value;
	return *this;
}

template<>
inline Json& Json::operator=(std::map<std::string, Json> map) {
	dataType = DataType::OBJECT;
	object = map;
	return *this;
}

template <>
inline Json& Json::operator=(const char* value) {
	dataType = DataType::STRING;
	string = std::string(value);
	return *this;
}

/* -------------------- *
*		FROM JSON		*
* --------------------- */

template <typename T>
inline Json::operator T() const {
	static_assert(ReflectionData<T>::defined, "Use the JSON_C and JSON_M macros to define the members for type T.");

	ASSERT(dataType == DataType::OBJECT);
	T res;

	constexpr size_t memberCount = std::tuple_size<decltype(Json::ReflectionData<T>::properties)>::value;
	for_sequence(std::make_index_sequence<memberCount>{}, [&](auto i) {
		constexpr auto property = std::get<i>(Json::ReflectionData<T>::properties);
		using Type = typename decltype(property)::Type;

		if constexpr (!std::is_array_v<Type>) {
			if (object.find(property.name) != object.end()) {
				Type val = (*this)[property.name];
				res.*(property.member) = val;
			}
		}
		else
			Json::setArray((*this)[property.name], res.*(property.member), sizeof(Type) / Json::array_size<Type>, Json::array_size<Type>);
		});

	return res;
}

template <typename T>
	requires(std::_Is_specialization_v<T, std::vector>)
inline Json::operator T() const {
	ASSERT(dataType == DataType::ARRAY);

	T vector(array.size());
	for (size_t i = 0; i < array.size(); i++) {
		using Type = decltype(vector)::value_type;
		Type value = array.at(i);
		vector.at(i) = value;
	}
	return vector;
}

template <typename T>
	requires(std::is_arithmetic_v<T> or std::is_enum_v<T>)
inline Json::operator T() const {
	ASSERT(dataType == DataType::NUMBER);
	if constexpr (std::is_enum_v<T>)
		return static_cast<T>((int)number);
	else
		return number;
}

inline Json::operator std::string() const {
	ASSERT(dataType == DataType::STRING);
	return string;
}

inline Json::operator std::map<std::string, Json>() const {
	ASSERT(dataType == DataType::OBJECT);
	return object;
}

/* ---------------------------- *
*		TYPELESS CONVERSION		*
* ----------------------------- */

template <typename T>
inline static Json::converter_function_t* Json::GetConverter() {
	return Json::anyToJson<T>;
}

template <typename T>
inline static Json::parser_function_t* Json::GetParser() {
	return Json::jsonToAny<T>;
}

template <typename T>
inline static Json Json::anyToJson(const unsigned char* value) {
	return (Json) * ((T*)value);
}

template <typename T>
inline static void Json::jsonToAny(const Json& value, void* dst) {
	*((T*)dst) = value;
}

template <>
inline static void Json::jsonToAny<std::string>(const Json& value, void* dst) {
	*((std::string*)dst) = value.operator std::string();
}

/* ------------------------ *
 *		INPUT & OUTPUT		*
 * ------------------------ */

inline void printIndent(std::ostream& os, unsigned indent) {
	for (unsigned i = 0; i < indent; i++)
		os << "   ";
}

inline std::ostream& operator<<(std::ostream& os, const Json& object) {
	static unsigned indent = 0;

	using enum Json::DataType;
	switch (object.dataType)
	{
	case NUMBER: os << std::format("{}", object.number); break;
	case STRING: os << "\"" << object.string << "\""; break;
	case ARRAY: {
		os << "[";
		if (object.array.size() > 0 && object.array.at(0).dataType != Json::DataType::NUMBER) {
			if (Json::s_indentOutput) { os << "\n"; printIndent(os, ++indent); }
		}
		for (size_t i = 0; i < object.array.size(); ++i) {
			os << object.array.at(i);
			if (i != object.array.size() - 1) {
				os << ",";
				if (object.array.size() > 0 && object.array.at(0).dataType != Json::DataType::NUMBER) {
					if (Json::s_indentOutput) { os << "\n"; printIndent(os, indent); }
				}
			}
		}
		if (object.array.size() > 0 && object.array.at(0).dataType != Json::DataType::NUMBER) {
			if (Json::s_indentOutput) { os << "\n"; printIndent(os, --indent); }
		}
		os << "]";
	} break;
	case OBJECT: {
		os << "{";
		if (Json::s_indentOutput) { os << "\n"; printIndent(os, ++indent); }
		int i = 0;
		for (auto& member : object.object) {
			os << "\"" << member.first << "\"" << ":" << member.second;
			if (i != object.object.size() - 1) {
				os << ",";
				if (Json::s_indentOutput) { os << "\n"; printIndent(os, indent); }
			}
			++i;
		}
		if (Json::s_indentOutput) { os << "\n"; printIndent(os, --indent); }
		os << "}";
	} break;
	default:
		break;
	}
	return os;
}


inline Json read(std::istream& is) {
	Json object;

	char firstC;
	is >> std::skipws >> firstC;

	if (firstC == '{') {
		object.dataType = Json::DataType::OBJECT;
		while (true) {
			char c;
			is >> std::skipws >> c;
			if (c == '}') {
				return object;
			}
			if (c == ',') {
				continue;
			}
			if (c == '"') {
				std::string name;
				char c;
				do {
					is >> std::noskipws >> c;
					name += c;
				} while (c != '"');
				name.erase(name.length() - 1);

				is >> std::skipws >> c;
				if (c != ':')
					throw;

				object[name] = read(is);

				continue;
			}
			throw;
		}
	}
	if (firstC == '[') {
		object.dataType = Json::DataType::ARRAY;

		while (true) {
			char c;
			int backtrack = is.tellg();
			is >> std::skipws >> c;
			if (c == ',') {
				continue;
			}
			if (c == ']') {
				return object;
			}

			is.seekg(backtrack);
			object.array.push_back(read(is));
		}
	}
	if (firstC >= '0' && firstC <= '9' || firstC == '-') {
		object.dataType = Json::DataType::NUMBER;

		int backtrack = is.tellg();
		std::string data;
		data += firstC;
		char c;
		do {
			backtrack = is.tellg();
			is >> c;
			data += c;
		} while (c >= '0' && c <= '9' || c == '.' || c == 'e' || c == '-' || c == '+');

		is.seekg(backtrack);
		object.number = std::stold(data);
		return object;
	}
	if (firstC == '"') {
		object.dataType = Json::DataType::STRING;
		std::string data;
		char c;
		is >> std::noskipws;
		do {
			is >> c;
			data += c;
		} while (c != '"');

		data.erase(data.length() - 1);
		object.string = data;
		is >> std::skipws;
		return object;
	}

	throw;
}

inline std::istream& operator>>(std::istream& is, Json& object) {
	object = read(is);
	return is;
}

/* ---------------- *
*		WRAPPER		*
* ----------------- */

template <typename T>
std::ostream& operator<<(std::ostream& os, const Json::wrap<T>& object) {
	os << (Json)(T)object;
	return os;
}

template <typename T>
std::istream& operator>>(std::istream& is, Json::wrap<T>& object) {
	Json input;
	is >> input;
	((T&)object) = input;
	return is;
}
