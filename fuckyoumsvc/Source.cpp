#include <concepts>
#ifdef _MSVC_LANG
#endif
template<typename T> 
requires std::is_same
struct stupidmsvc {
	T print() {
		return T{};
	}
};

int main() {
	stupidmsvc<size_t>().print();
	return 0;
}