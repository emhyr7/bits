#include <stdio.h>
#include <immintrin.h>
#include <stdlib.h>
#include <memory.h>

#if defined(_WIN32)

#include <intrin.h>

extern int __stdcall QueryPerformanceCounter  (unsigned long long *);
extern int __stdcall QueryPerformanceFrequency(unsigned long long *);

extern void *__stdcall VirtualAlloc(void *, unsigned long long, unsigned, unsigned);

#endif

#if defined(__clang__) || defined(__GNUC__)
#define TRAP() __builtin_debugtrap()
#elif defined(_MSC_VER)
#define TRAP() __debugbreak()
#endif

#define Assert(x) do { if (!(x)) TRAP(); } while (0)

typedef long long          Bits64;
typedef unsigned long long Size;
typedef unsigned long      Index;
typedef unsigned long      Count;
typedef int                Boolean;
typedef long long          Word;

typedef struct {
	Bits64 *pointer;
	Index   index;
} BitLocation;

#define LMASK(x) (~(-1ll << (x)))

typedef Word Flags;

#if defined(__clang__) || defined(__GNUC__)
#define _ffsll(x) __builtin_ffsll(x)
#else
inline int _ffsll(long long int x) {
	int r;
	if (!_BitScanForward64(x)) r = -1;
	return r + 1;
}
#endif

/* NOTE(Emhyr): "reverse" means to decrement the word pointer. it doesn't scan the bits reversely */

static inline BitLocation DoFindBit(Boolean reverse, Boolean toClear, Bits64 *p, Bits64 *q) {
	BitLocation result;
	__mmask8 m;
	__m512i v, cmp;
	
	cmp = _mm512_set1_epi64(toClear ? -1 : 0);
	if (reverse) p -= 7;
	for (;;) {
		if (p == q) break;

		v = _mm512_load_epi64(p);
		m = _mm512_cmpneq_epu64_mask(v, cmp);
		if (m) break;

		long long inc = sizeof(v) / sizeof(*p);
		if (reverse) inc = -inc;
		p += inc;
	}

	int zcnt;
	if (reverse) zcnt = 7 - _lzcnt_u32((unsigned)m << 24);
	else zcnt = _tzcnt_u32(m);
	result.pointer = p + zcnt;
	if (result.pointer != q) {
		Bits64 x = *result.pointer;
		if (toClear) x = ~x;
		result.index = _ffsll(x) - 1;
	}
	return result;
}

BitLocation FindSetBit         (Bits64 *p, Bits64 *q) { return DoFindBit(0, 0, p, q); }
BitLocation FindClearBit       (Bits64 *p, Bits64 *q) { return DoFindBit(0, 1, p, q); }
BitLocation ReverseFindSetBit  (Bits64 *p, Bits64 *q) { return DoFindBit(1, 0, p, q); }
BitLocation ReverseFindClearBit(Bits64 *p, Bits64 *q) { return DoFindBit(1, 1, p, q); }

#define WIDTHOF(x) (sizeof(x) * 8)

/* NOTE(Emhyr): if n <= half a word size, we can buffer-shift bits into a word to
compare with a bitmask. */

BitLocation DoFindBits(Boolean reverse, Boolean toClear, Size n, Bits64 *p, Bits64 *q) {
	if (n == 1) {
		if (reverse) {
			if (toClear) return ReverseFindClearBit(p, q);
			else return ReverseFindSetBit(p, q);
		} else {
			if (toClear) return FindClearBit(p, q);
			else return FindSetBit(p, q);
		}
	}
	
	BitLocation result;
	Size c, i, j, z;
	Bits64 w;
	
	result.pointer = p;
	result.index = 0;
	c = 0;
	for (; p != q; reverse ? --p : ++p) {
		w = *p;
		if (toClear) w = ~w;
	retry:
		result.pointer = p;
		i = _ffsll(w);
		if (!i) continue;
		result.index = i - 1;
		w |= ~(-1ll << (i - 1));
		j = _ffsll(~w);
		if (!j) j = WIDTHOF(*p) + 1;
		c += j - i;
		if (c >= n) break;
		w &= -1ll << (j - 1);
		if (!w) continue;
		if (j == WIDTHOF(*p) + 1) {
			do {
				w = *++p;
				if (!toClear) w = ~w;
				if (p == q) goto finished;
				z = _tzcnt_u64(w);
				c += z;
				if (c >= n) goto finished;
			} while (z == WIDTHOF(*p));
			w |= ~(-1ll << (z - 1));
			w = ~w;
		}
		c = 0;
		goto retry;
	}
finished:
	if (p == q) result = (BitLocation){0, 0};
	return result;
}

BitLocation FindSetBits         (Size n, Bits64 *p, Bits64 *q) { return DoFindBits(0, 0, n, p, q); }
BitLocation FindClearBits       (Size n, Bits64 *p, Bits64 *q) { return DoFindBits(0, 1, n, p, q); }
BitLocation ReverseFindSetBits  (Size n, Bits64 *p, Bits64 *q) { return DoFindBits(1, 0, n, p, q); }
BitLocation ReverseFindClearBits(Size n, Bits64 *p, Bits64 *q) { return DoFindBits(1, 1, n, p, q); }

BitLocation correct;

size_t ClockRate;
size_t Start, End;

#define BeginClock() (void)QueryPerformanceCounter(&Start)
#define EndClock()   (void)QueryPerformanceCounter(&End)
#define Elapse()     ((End - Start) * 1000000000 / ClockRate)

#define N 1000000
_Alignas(__m512i) Bits64 p[N];
Bits64 *q = p + N;

int main(int argc, char *argv[]) {
	QueryPerformanceFrequency(&ClockRate);
	memset(p, -1, N * sizeof(Bits64));
	unsigned long long random = 128;
	_rdrand64_step(&random);
	correct.pointer = p + random % N;
	memset(correct.pointer, random, sizeof(Bits64));
	Assert(_BitScanForward64(&correct.index, ~*correct.pointer));
	printf("random %% N: %llu\n", random % N);
	printf("random: %llu\n", random);
	printf("correct.pointer: %p -> %llu\n", correct.pointer, *correct.pointer);
	printf("correct.index: %lu\n", correct.index);
	printf("q: %p\n", q);

	printf("\n\n");

	BitLocation result;

	printf("FindClearBit:\n");
	QueryPerformanceCounter(&Start);
	result = FindClearBit(p, q);
	QueryPerformanceCounter(&End);
	printf("\telapse: %llu\n", Elapse());
	printf("\tresult: %p -> %llu & %lu\n", result.pointer, *result.pointer, result.index);
	
	printf("ReverseFindClearBit:\n");
	QueryPerformanceCounter(&Start);
	result = ReverseFindClearBit(q - 1, p - 1);
	QueryPerformanceCounter(&End);
	printf("\telapse: %llu\n", Elapse());
	printf("\tresult: %p -> %llu & %lu\n", result.pointer, *result.pointer, result.index);
	
	printf("FindClearBits:\n");
	QueryPerformanceCounter(&Start);
	result = FindClearBits(1, p, q);
	QueryPerformanceCounter(&End);
	printf("\telapse: %llu\n", Elapse());
	printf("\tresult: %p -> %llu & %lu\n", result.pointer, result.pointer ? *result.pointer : 0, result.index);
	
	printf("FindClearBits:\n");
	QueryPerformanceCounter(&Start);
	result = ReverseFindClearBits(1, q - 1, p - 1);
	QueryPerformanceCounter(&End);
	printf("\telapse: %llu\n", Elapse());
	printf("\tresult: %p -> %llu & %lu\n", result.pointer, result.pointer ? *result.pointer : 0, result.index);
}
